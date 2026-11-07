#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_CONTENTHASH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_CONTENTHASH_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__GNUC__) && defined(__x86_64__)
#include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

// 128-bit content hash for "are these bytes still the bytes we saw" checks that would otherwise keep a
// byte copy (texture snapshots). The construction is XXH3's long-input path (64-byte stripes into
// eight 64-bit accumulators, a scramble per 1 KiB block, 128-bit merge) with a secret derived from
// splitmix64; the 64-bit lanes take the 32x32 products SSE2 computes two at a time.
namespace AgcDriver::ContentHash {

struct Hash128 {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
    bool operator==(const Hash128&) const = default;
};

namespace Detail {

inline constexpr std::uint64_t Prime32_1 = 0x9E3779B1u;
inline constexpr std::uint64_t Prime32_2 = 0x85EBCA77u;
inline constexpr std::uint64_t Prime32_3 = 0xC2B2AE3Du;
inline constexpr std::uint64_t Prime64_1 = 0x9E3779B185EBCA87ull;
inline constexpr std::uint64_t Prime64_2 = 0xC2B2AE3D27D4EB4Full;
inline constexpr std::uint64_t Prime64_3 = 0x165667B19E3779F9ull;
inline constexpr std::uint64_t Prime64_4 = 0x85EBCA77C2B2AE63ull;
inline constexpr std::uint64_t Prime64_5 = 0x27D4EB2F165667C5ull;
inline constexpr std::size_t SecretBytes = 192;
inline constexpr std::size_t StripeBytes = 64;
inline constexpr std::size_t StripesPerBlock = (SecretBytes - StripeBytes) / 8;
inline constexpr std::size_t BlockBytes = StripesPerBlock * StripeBytes;

constexpr std::array<unsigned char, SecretBytes> MakeSecret() {
    std::array<unsigned char, SecretBytes> secret{};
    std::uint64_t state = 0x41505335c0ffee11ull;
    for (std::size_t i = 0; i < SecretBytes; i += 8) {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        for (std::size_t b = 0; b < 8; ++b) secret[i + b] = static_cast<unsigned char>(z >> (8 * b));
    }
    return secret;
}

inline constexpr std::array<unsigned char, SecretBytes> Secret = MakeSecret();

inline std::uint64_t Read64(const unsigned char* at) noexcept {
    std::uint64_t value;
    std::memcpy(&value, at, sizeof(value));
    return value;
}

inline std::uint64_t Mul128Fold64(std::uint64_t a, std::uint64_t b) noexcept {
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    return static_cast<std::uint64_t>(product) ^ static_cast<std::uint64_t>(product >> 64);
}

inline std::uint64_t Avalanche(std::uint64_t h) noexcept {
    h ^= h >> 37;
    h *= 0x165667919E3779F9ull;
    return h ^ (h >> 32);
}

#if defined(__SSE2__) || defined(_M_X64)
struct Accumulators {
    __m128i lanes[4];
    Accumulators() noexcept {
        lanes[0] = _mm_set_epi64x(static_cast<long long>(Prime64_1), static_cast<long long>(Prime32_3));
        lanes[1] = _mm_set_epi64x(static_cast<long long>(Prime64_3), static_cast<long long>(Prime64_2));
        lanes[2] = _mm_set_epi64x(static_cast<long long>(Prime32_2), static_cast<long long>(Prime64_4));
        lanes[3] = _mm_set_epi64x(static_cast<long long>(Prime32_1), static_cast<long long>(Prime64_5));
    }
    void Accumulate(const unsigned char* input, const unsigned char* secret) noexcept {
        for (int i = 0; i < 4; ++i) {
            const __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input) + i);
            const __m128i key = _mm_loadu_si128(reinterpret_cast<const __m128i*>(secret) + i);
            const __m128i mixed = _mm_xor_si128(data, key);
            const __m128i product = _mm_mul_epu32(mixed, _mm_shuffle_epi32(mixed, _MM_SHUFFLE(0, 3, 0, 1)));
            const __m128i swapped = _mm_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2));
            lanes[i] = _mm_add_epi64(product, _mm_add_epi64(lanes[i], swapped));
        }
    }
    void Scramble(const unsigned char* secret) noexcept {
        const __m128i prime = _mm_set1_epi32(static_cast<int>(Prime32_1));
        for (int i = 0; i < 4; ++i) {
            __m128i value = lanes[i];
            value = _mm_xor_si128(value, _mm_srli_epi64(value, 47));
            value = _mm_xor_si128(value, _mm_loadu_si128(reinterpret_cast<const __m128i*>(secret) + i));
            const __m128i low = _mm_mul_epu32(value, prime);
            const __m128i high = _mm_mul_epu32(_mm_srli_epi64(value, 32), prime);
            lanes[i] = _mm_add_epi64(low, _mm_slli_epi64(high, 32));
        }
    }
    void Store(std::uint64_t (&out)[8]) const noexcept {
        for (int i = 0; i < 4; ++i) _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 2 * i), lanes[i]);
    }
};
#else
struct Accumulators {
    std::uint64_t lanes[8] = {Prime32_3, Prime64_1, Prime64_2, Prime64_3, Prime64_4, Prime32_2, Prime64_5, Prime32_1};
    void Accumulate(const unsigned char* input, const unsigned char* secret) noexcept {
        for (int i = 0; i < 8; ++i) {
            const auto data = Read64(input + 8 * i);
            const auto mixed = data ^ Read64(secret + 8 * i);
            lanes[i ^ 1] += data;
            lanes[i] += (mixed & 0xffffffffu) * (mixed >> 32);
        }
    }
    void Scramble(const unsigned char* secret) noexcept {
        for (int i = 0; i < 8; ++i) {
            auto value = lanes[i];
            value ^= value >> 47;
            value ^= Read64(secret + 8 * i);
            lanes[i] = value * Prime32_1;
        }
    }
    void Store(std::uint64_t (&out)[8]) const noexcept {
        for (int i = 0; i < 8; ++i) out[i] = lanes[i];
    }
};
#endif

inline std::uint64_t Merge(const std::uint64_t (&lanes)[8], const unsigned char* secret, std::uint64_t start) noexcept {
    auto result = start;
    for (int i = 0; i < 4; ++i) result += Mul128Fold64(lanes[2 * i] ^ Read64(secret + 16 * i), lanes[2 * i + 1] ^ Read64(secret + 16 * i + 8));
    return Avalanche(result);
}

#if defined(__GNUC__) && defined(__x86_64__)
__attribute__((target("avx2"))) inline Hash128 OfAvx2(const unsigned char* input, std::size_t bytes) noexcept {
    const auto* secret = Secret.data();
    __m256i a = _mm256_set_epi64x(static_cast<long long>(Prime64_3), static_cast<long long>(Prime64_2), static_cast<long long>(Prime64_1), static_cast<long long>(Prime32_3));
    __m256i b = _mm256_set_epi64x(static_cast<long long>(Prime32_1), static_cast<long long>(Prime64_5), static_cast<long long>(Prime32_2), static_cast<long long>(Prime64_4));
    const auto accumulate = [&] (const unsigned char* p, const unsigned char* k) __attribute__((target("avx2"))) {
        const __m256i d0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i d1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
        const __m256i m0 = _mm256_xor_si256(d0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k)));
        const __m256i m1 = _mm256_xor_si256(d1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 32)));
        a = _mm256_add_epi64(a, _mm256_add_epi64(_mm256_shuffle_epi32(d0, _MM_SHUFFLE(1, 0, 3, 2)), _mm256_mul_epu32(m0, _mm256_srli_epi64(m0, 32))));
        b = _mm256_add_epi64(b, _mm256_add_epi64(_mm256_shuffle_epi32(d1, _MM_SHUFFLE(1, 0, 3, 2)), _mm256_mul_epu32(m1, _mm256_srli_epi64(m1, 32))));
    };
    const auto scramble = [&] (__m256i value, const unsigned char* k) __attribute__((target("avx2"))) {
        value = _mm256_xor_si256(value, _mm256_srli_epi64(value, 47));
        value = _mm256_xor_si256(value, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k)));
        const __m256i prime = _mm256_set1_epi32(static_cast<int>(Prime32_1));
        return _mm256_add_epi64(_mm256_mul_epu32(value, prime), _mm256_slli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(value, 32), prime), 32));
    };
    const std::size_t blocks = (bytes - 1) / BlockBytes;
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto* at = input + block * BlockBytes;
        for (std::size_t stripe = 0; stripe < StripesPerBlock; ++stripe) accumulate(at + stripe * StripeBytes, secret + stripe * 8);
        a = scramble(a, secret + SecretBytes - StripeBytes);
        b = scramble(b, secret + SecretBytes - StripeBytes + 32);
    }
    const std::size_t stripes = ((bytes - 1) - blocks * BlockBytes) / StripeBytes;
    const auto* tail = input + blocks * BlockBytes;
    for (std::size_t stripe = 0; stripe < stripes; ++stripe) accumulate(tail + stripe * StripeBytes, secret + stripe * 8);
    accumulate(input + bytes - StripeBytes, secret + SecretBytes - StripeBytes - 7);
    std::uint64_t lanes[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), a);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes + 4), b);
    const auto size = static_cast<std::uint64_t>(bytes);
    return {Merge(lanes, secret + 11, size * Prime64_1), Merge(lanes, secret + SecretBytes - StripeBytes - 11, ~(size * Prime64_2))};
}
#endif

}

// Hash of `bytes` bytes at `data` (the length is part of the hash).
inline Hash128 Of(const void* data, std::size_t bytes) noexcept {
    using namespace Detail;
#if defined(__GNUC__) && defined(__x86_64__)
    if (bytes >= BlockBytes && __builtin_cpu_supports("avx2")) return OfAvx2(static_cast<const unsigned char*>(data), bytes);
#endif
    const auto* secret = Secret.data();
    const auto* input = static_cast<const unsigned char*>(data);
    unsigned char padded[StripeBytes];
    if (bytes < StripeBytes) {
        std::memset(padded, 0, sizeof(padded));
        if (bytes != 0) std::memcpy(padded, input, bytes);
        input = padded;
    }
    const std::size_t length = bytes < StripeBytes ? StripeBytes : bytes;
    Accumulators accumulators;
    const std::size_t blocks = (length - 1) / BlockBytes;
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto* at = input + block * BlockBytes;
        for (std::size_t stripe = 0; stripe < StripesPerBlock; ++stripe) accumulators.Accumulate(at + stripe * StripeBytes, secret + stripe * 8);
        accumulators.Scramble(secret + SecretBytes - StripeBytes);
    }
    const std::size_t stripes = ((length - 1) - blocks * BlockBytes) / StripeBytes;
    const auto* tail = input + blocks * BlockBytes;
    for (std::size_t stripe = 0; stripe < stripes; ++stripe) accumulators.Accumulate(tail + stripe * StripeBytes, secret + stripe * 8);
    accumulators.Accumulate(input + length - StripeBytes, secret + SecretBytes - StripeBytes - 7);
    std::uint64_t lanes[8];
    accumulators.Store(lanes);
    const auto size = static_cast<std::uint64_t>(bytes);
    return {Merge(lanes, secret + 11, size * Prime64_1), Merge(lanes, secret + SecretBytes - StripeBytes - 11, ~(size * Prime64_2))};
}

}

#endif
