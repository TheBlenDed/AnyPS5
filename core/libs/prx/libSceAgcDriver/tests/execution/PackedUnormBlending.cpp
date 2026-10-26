#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::ColorTargetLayout;
using AgcDriver::Graphics::ColorTileMode;
using AgcDriver::Graphics::Require;
using ShaderRecompiler::ColorTargetBlend;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 128;
constexpr std::uint32_t ScissorWidth = 192;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Unorm10_11_11Info = 0x8018u;
constexpr std::uint32_t TiledAttrib3 = 0x4dc6c000u;
constexpr std::uint32_t LinearAttrib3 = 0x09000000u;
constexpr std::uint32_t Float32Export = 9;
constexpr std::uint32_t Float16Export = 4;
constexpr std::uint32_t PixelInterlockCapability = 5378;
constexpr std::uint32_t KeepYBelowX = 0x7c220101u;
constexpr std::uint32_t Overlaps = 8;
constexpr std::array<std::uint32_t, 3> ChannelBits{11, 11, 10};
constexpr std::array<std::uint32_t, 3> ChannelOffsets{0, 11, 22};

constexpr ColorTargetBlend AlphaBlend{true, 4, 5, 0, 7};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) std::array<std::array<std::uint32_t, 64>, 256> Programs{};
std::size_t programCount = 0;

const std::vector<std::array<float, 4>>& Triangles(std::uint32_t count) {
    static std::array<std::vector<std::array<float, 4>>, Overlaps + 1> cache;
    auto& vertices = cache.at(count);
    if (vertices.empty()) {
        for (std::uint32_t triangle = 0; triangle < count; ++triangle) {
            vertices.push_back({-1.0f, -1.0f, 0.5f, 1.0f});
            vertices.push_back({3.0f, -1.0f, 0.5f, 1.0f});
            vertices.push_back({-1.0f, 3.0f, 0.5f, 1.0f});
        }
    }
    return vertices;
}

float ClampUnit(float value) {
    return value > 0.0f ? (value < 1.0f ? value : 1.0f) : 0.0f;
}

std::uint32_t Unorm(float value, std::uint32_t bits) {
    const auto maximum = (1u << bits) - 1u;
    if (!(value > 0.0f)) return 0;
    if (value >= 1.0f) return maximum;
    return static_cast<std::uint32_t>(std::nearbyint(static_cast<double>(value) * maximum));
}

float Factor(std::uint32_t factor, float source, float alpha, float destination) {
    volatile float one = 1.0f;
    switch (factor) {
        case 0: return 0.0f;
        case 1: return 1.0f;
        case 2: return source;
        case 3: return one - source;
        case 4: return alpha;
        case 5: return one - alpha;
        case 8: return destination;
        case 9: return one - destination;
        default: throw std::runtime_error("the test uses an unsupported blend factor");
    }
}

std::uint32_t Blend(std::uint32_t word, const std::array<float, 4>& color, const ColorTargetBlend& blend) {
    const float alpha = ClampUnit(color[3]);
    std::uint32_t result = 0;
    for (std::uint32_t channel = 0; channel < 3; ++channel) {
        const auto bits = ChannelBits[channel];
        const auto maximum = (1u << bits) - 1u;
        auto field = (word >> ChannelOffsets[channel]) & maximum;
        if (((blend.writeMask >> channel) & 1u) != 0) {
            const float source = ClampUnit(color[channel]);
            const float destination = static_cast<float>(field) / static_cast<float>(maximum);
            volatile float sourceTerm = source * Factor(blend.sourceFactor, source, alpha, destination);
            volatile float destinationTerm = destination * Factor(blend.destinationFactor, source, alpha, destination);
            const float blended = blend.function == 0 ? sourceTerm + destinationTerm : blend.function == 1 ? sourceTerm - destinationTerm : destinationTerm - sourceTerm;
            field = Unorm(blended, bits);
        }
        result |= field << ChannelOffsets[channel];
    }
    return result;
}

std::uint32_t Half(float value) {
    Require(value == 0.0f || (std::fabs(value) >= 0.0625f && std::fabs(value) <= 2.0f), "the test's half values are normal and at most 2 in magnitude");
    if (value == 0.0f) return 0;
    const auto bits = std::bit_cast<std::uint32_t>(value);
    Require((bits & 0x1fffu) == 0, "the test's half values are exact");
    return ((bits >> 16u) & 0x8000u) | ((((bits >> 23u) & 0xffu) - 112u) << 10u) | ((bits >> 13u) & 0x3ffu);
}

std::span<const std::uint32_t> Store(const std::vector<std::uint32_t>& words) {
    Require(programCount < Programs.size() && words.size() <= Programs[0].size(), "too many test programs");
    auto& code = Programs[programCount++];
    std::copy(words.begin(), words.end(), code.begin());
    return std::span<const std::uint32_t>(code.data(), words.size());
}

void MoveLiteral(std::vector<std::uint32_t>& code, std::uint32_t vgpr, std::uint32_t value) {
    code.push_back(0x7e0002ffu | (vgpr << 17u));
    code.push_back(value);
}

void MultiplyLiteral(std::vector<std::uint32_t>& code, std::uint32_t vgpr, std::uint32_t source, float value) {
    code.push_back((0x08u << 25u) | (vgpr << 17u) | (source << 9u) | 0xffu);
    code.push_back(std::bit_cast<std::uint32_t>(value));
}

std::span<const std::uint32_t> ConstantProgram(const std::array<float, 4>& color, std::uint32_t target = 0u, std::uint32_t additionalTarget = 8u) {
    std::vector<std::uint32_t> code;
    for (std::uint32_t lane = 0; lane < 4; ++lane) MoveLiteral(code, 4 + lane, std::bit_cast<std::uint32_t>(color[lane]));
    if (additionalTarget < 8u) code.insert(code.end(), {0xf800000fu | (additionalTarget << 4u), 0x07060504u});
    code.insert(code.end(), {0xf800180fu | (target << 4u), 0x07060504u, 0xbf810000u});
    return Store(code);
}

std::span<const std::uint32_t> HalfProgram(const std::array<float, 4>& color) {
    std::vector<std::uint32_t> code;
    MoveLiteral(code, 4, Half(color[0]) | (Half(color[1]) << 16u));
    MoveLiteral(code, 5, Half(color[2]) | (Half(color[3]) << 16u));
    code.insert(code.end(), {0xf8001c0fu, 0x00000504u, 0xbf810000u});
    return Store(code);
}

constexpr std::array<float, 4> Slopes{1.0f / 200.0f, 1.0f / 256.0f, 1.0f / 100.0f, 1.0f / 128.0f};

std::span<const std::uint32_t> VaryingProgram() {
    std::vector<std::uint32_t> code;
    for (std::uint32_t lane = 0; lane < 4; ++lane) MultiplyLiteral(code, 4 + lane, lane < 2 ? 0u : 1u, Slopes[lane]);
    code.insert(code.end(), {0xf800180fu, 0x07060504u, 0xbf810000u});
    return Store(code);
}

std::array<float, 4> VaryingColor(std::uint32_t x, std::uint32_t y) {
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;
    return {px * Slopes[0], px * Slopes[1], py * Slopes[2], py * Slopes[3]};
}

std::span<const std::uint32_t> KillingProgram(const std::array<float, 4>& color, bool exportFirst) {
    std::vector<std::uint32_t> code;
    for (std::uint32_t lane = 0; lane < 4; ++lane) MoveLiteral(code, 4 + lane, std::bit_cast<std::uint32_t>(color[lane]));
    if (exportFirst) {
        code.insert(code.end(), {0xf800000fu, 0x07060504u, KeepYBelowX, 0xf8001890u, 0x00000000u, 0xbf810000u});
    } else {
        code.insert(code.end(), {KeepYBelowX, 0xf800180fu, 0x07060504u, 0xbf810000u});
    }
    return Store(code);
}

std::span<const std::uint32_t> MixedProgram(const std::array<float, 4>& blended, const std::array<float, 4>& attached) {
    std::vector<std::uint32_t> code;
    for (std::uint32_t lane = 0; lane < 4; ++lane) MoveLiteral(code, 4 + lane, std::bit_cast<std::uint32_t>(blended[lane]));
    for (std::uint32_t lane = 0; lane < 4; ++lane) MoveLiteral(code, 8 + lane, std::bit_cast<std::uint32_t>(attached[lane]));
    code.insert(code.end(), {0xf800000fu, 0x07060504u, 0xf800181fu, 0x0b0a0908u, 0xbf810000u});
    return Store(code);
}

struct Block {
    explicit Block(std::size_t bytes) : bytes(bytes), watched(AgcDriver::GuestMemory::WriteWatched()) {
#ifdef _WIN32
        if (watched) {
            data = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, BlockBytes));
            GuestArena::GuestArenaCommit_nid_postfix(data, bytes, PAGE_READWRITE, bytes);
        } else {
            data = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        }
#else
        if (watched) {
            void* raw = mmap(nullptr, bytes + BlockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            Require(raw != MAP_FAILED, "cannot map the color target");
            const auto begin = reinterpret_cast<std::uintptr_t>(raw);
            const auto aligned = (begin + BlockBytes - 1) & ~(static_cast<std::uintptr_t>(BlockBytes) - 1);
            if (aligned != begin) munmap(raw, aligned - begin);
            if (aligned + bytes != begin + bytes + BlockBytes) munmap(reinterpret_cast<void*>(aligned + bytes), begin + BlockBytes - aligned);
            data = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(data, bytes);
        } else {
            data = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, bytes));
        }
#endif
        Require(data != nullptr, "cannot allocate the color target");
        Require(!watched || AgcDriver::GuestMemory::Watched(Address(), bytes), "the color target is not write-watched");
        GuestAllocations::Mutation mutation;
        mutation.Add(data, bytes, true, true, true);
    }
    ~Block() {
        AgcDriver::Graphics::StorageTexture::FlushPending(Address(), bytes, nullptr, "test release");
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(data);
        }
#ifdef _WIN32
        if (watched) {
            GuestArena::GuestArenaReset_nid_postfix(data, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(data, bytes);
        } else {
            VirtualFree(data, 0, MEM_RELEASE);
        }
#else
        if (watched) {
            munmap(data, bytes);
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(data, bytes);
        } else {
            std::free(data);
        }
#endif
    }
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::size_t bytes;
    bool watched;
    std::uint8_t* data = nullptr;
};

AgcDriver::Graphics::ColorTarget DecodeTarget(const Block& block, std::uint32_t attrib3) {
    AgcDriver::Registers cx;
    cx[0x318] = static_cast<std::uint32_t>(block.Address() >> 8u);
    cx[0x31b] = 0;
    cx[0x31c] = Unorm10_11_11Info;
    cx[0x31d] = 0;
    cx[0x3b0] = ((Width - 1u) << 14u) | (Height - 1u);
    cx[0x3b8] = attrib3;
    cx[0x390] = static_cast<std::uint32_t>(block.Address() >> 40u);
    auto color = AgcDriver::Graphics::DecodeColorBuffer(cx, 0);
    Require(color.format == VK_FORMAT_R32_UINT && color.packing == ShaderRecompiler::ColorExportPacking::Unorm10_11_11 && color.bytes == block.bytes, "the 10_11_11 unorm target did not decode as a packed R32_UINT surface");
    return color;
}

struct Attachment {
    alignas(256) std::array<std::uint8_t, Width * Height * 4> pixels{};
};

Attachment Attached;

void Draw(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::ColorTarget& color, const ColorTargetBlend& blend, std::uint32_t exportFormat, std::span<const std::uint32_t> pixelCode, std::uint32_t triangles = 1, Attachment* attached = nullptr, std::uint32_t exportIndex = 0u, const AgcDriver::Graphics::ColorTarget* additionalTarget = nullptr) {
    const auto target = device.Target();
    constexpr std::uint32_t waveSize = 64;
    const auto& vertices = Triangles(triangles);
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto address = reinterpret_cast<std::uintptr_t>(vertices.data());
    const std::array<std::uint32_t, 4> vertexBuffer{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(vertices.size()), 0x01016facu};
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {waveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, waveSize, waveSize, std::nullopt, std::nullopt};
    auto blended = color;
    blended.exportIndex = exportIndex;
    state.blendedColors = {blended};
    state.targetBlends[exportIndex] = blend;
    if (additionalTarget != nullptr) {
        state.blendedColors.push_back(*additionalTarget);
        state.targetBlends[additionalTarget->exportIndex] = blend;
    }
    state.color = blended;
    if (attached != nullptr) {
        AgcDriver::Graphics::ColorTarget other{reinterpret_cast<std::uintptr_t>(attached->pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, attached->pixels.size(), 0xe4u};
        other.exportIndex = 1;
        other.slot = 1;
        state.colors = {other};
        VkPipelineColorBlendAttachmentState write{};
        write.colorWriteMask = 15;
        state.blends = {VkPipelineColorBlendAttachmentState{}, write};
        state.blend = write;
    }
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {ScissorWidth, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blendConstants = {};

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode.at(exportIndex) = static_cast<std::uint8_t>(exportFormat);
    if (additionalTarget != nullptr) pixel.targetOutputMode.at(additionalTarget->exportIndex) = static_cast<std::uint8_t>(exportFormat);
    if (attached != nullptr) pixel.targetOutputMode.at(1) = static_cast<std::uint8_t>(Float32Export);
    pixel.targetExportMapping = AgcDriver::Graphics::ExportMappings(state);
    pixel.targetExportPacking = AgcDriver::Graphics::ExportPackings(state);
    pixel.targetBlend = state.targetBlends;
    const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(pixelCode.data());
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{codeAddress, std::as_bytes(pixelCode)}}};
    auto registered = pixel;
    registered.targetExportMapping = {};
    registered.targetExportPacking = {};
    registered.targetBlend = {};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, codeAddress, pixelCode, 0, {}},
        {waveSize, 0, pixelUserData, std::nullopt, registered, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    const auto handle = ShaderRecompiler::PrepareShader(fragment);
    AgcDriver::DriverDetail::ShaderSnapshot snapshot{codeAddress, 0, 1, {pixelCode.begin(), pixelCode.end()}, {}};
    snapshot.prepared->entries.push_back({0, handle});
    fragment.shader.code = snapshot.code;
    fragment.context.pixel = pixel;
    const ShaderRecompiler::SrtRuntime runtime{pixelUserData, codeAddress};
    const auto invocation = AgcDriver::DriverDetail::InvocationFor(snapshot, 0, fragment);
    const auto pixelResult = invocation.Materialize(*invocation.Capture(runtime));
    Require(pixelResult->variantId == ShaderRecompiler::GetPreparedArtifact(*handle).variantId, "the blended draw did not use the registered pixel shader");
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, pixelResult.get(), PixelPushOffset(vertexPush, target)}
    }};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(vertices.size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
}

std::vector<std::uint32_t> ReadBack(AgcDriver::VulkanDevice& device, const Block& block) {
    AgcDriver::Graphics::StorageTexture::FlushPending(block.Address(), block.bytes, nullptr, "test read-back");
    device.WaitIdle();
    std::vector<std::uint32_t> stored(block.bytes / 4u);
    AgcDriver::GuestMemory::Read(block.Address(), std::as_writable_bytes(std::span(stored)), 1);
    return stored;
}

std::vector<std::uint32_t> Fill(Block& block, std::uint32_t seed) {
    std::vector<std::uint32_t> words(block.bytes / 4u);
    for (std::size_t index = 0; index < words.size(); ++index) words[index] = static_cast<std::uint32_t>((index + seed) * 0x9e3779b9u) ^ (seed << 7u);
    std::memcpy(block.data, words.data(), block.bytes);
    return words;
}

std::string Describe(const std::array<float, 4>& color, const ColorTargetBlend& blend) {
    char text[200];
    std::snprintf(text, sizeof(text), "(%.9g, %.9g, %.9g, alpha %.9g) with factors %u/%u, function %u, mask %u", color[0], color[1], color[2], color[3], blend.sourceFactor, blend.destinationFactor, blend.function, blend.writeMask);
    return text;
}

template <typename TColor, typename TKept>
void ExpectWhere(AgcDriver::VulkanDevice& device, const Block& block, const ColorTargetLayout& layout, const std::vector<std::uint32_t>& before, const ColorTargetBlend& blend, std::uint32_t blends, TColor&& color, TKept&& kept, const std::string& what) {
    const auto stored = ReadBack(device, block);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto index = layout.Offset(x, y) / 4u;
            auto expected = before[index];
            if (x < ScissorWidth && kept(x, y)) {
                for (std::uint32_t step = 0; step < blends; ++step) expected = Blend(expected, color(x, y), blend);
            }
            if (stored[index] == expected) continue;
            char detail[200];
            std::snprintf(detail, sizeof(detail), ": pixel (%u, %u) over 0x%08x holds 0x%08x, expected 0x%08x", x, y, before[index], stored[index], expected);
            throw std::runtime_error(what + detail);
        }
    }
}

template <typename TColor>
void Expect(AgcDriver::VulkanDevice& device, const Block& block, const ColorTargetLayout& layout, const std::vector<std::uint32_t>& before, const ColorTargetBlend& blend, std::uint32_t blends, TColor&& color, const std::string& what) {
    ExpectWhere(device, block, layout, before, blend, blends, color, [](std::uint32_t, std::uint32_t) { return true; }, what);
}

std::size_t TargetTests(AgcDriver::VulkanDevice& device, Block& block) {
    const ColorTargetLayout layout(Width, Height, ColorTileMode::RenderTarget, 4);
    Require(layout.Bytes() == block.bytes, "the test surface size disagrees with its layout");
    const auto color = DecodeTarget(block, TiledAttrib3);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    std::size_t draws = 0;
    std::uint32_t seed = 1;
    const auto constant = [&](const std::array<float, 4>& source, const ColorTargetBlend& blend, std::uint32_t triangles) {
        const auto before = Fill(block, seed++);
        Draw(device, color, blend, Float32Export, ConstantProgram(source), triangles);
        ++draws;
        Expect(device, block, layout, before, blend, triangles, [&](std::uint32_t, std::uint32_t) { return source; }, (triangles > 1 ? std::to_string(triangles) + " overlapping blends of " : std::string("a blend of ")) + Describe(source, blend));
    };
    const std::array<std::array<float, 4>, 9> sources{{
        {0.25f, 0.5f, 0.75f, 0.3f},
        {nan, -1.0f, 2.0f, 0.5f},
        {1.0f / 3.0f, 2.0f / 3.0f, 0.1f, 1.0f / 3.0f},
        {0.9f, 0.1f, 0.6f, 0.0f},
        {0.2f, 0.4f, 0.8f, 1.0f},
        {0.7f, infinity, 1e-30f, nan},
        {0.5f, 0.5f, 0.5f, 2.0f},
        {0.123f, 0.456f, 0.789f, -0.5f},
        {1023.5f / 2047.0f, 1.5f / 2047.0f, 511.5f / 1023.0f, 0.5f},
    }};
    for (const auto& source : sources) constant(source, AlphaBlend, 1);
    const std::array<ColorTargetBlend, 10> modes{{
        {true, 4, 5, 1, 7},
        {true, 4, 5, 4, 7},
        {true, 1, 1, 0, 7},
        {true, 8, 0, 0, 7},
        {true, 0, 2, 0, 7},
        {true, 9, 1, 0, 7},
        {true, 2, 3, 0, 7},
        {true, 1, 5, 0, 7},
        {true, 4, 5, 0, 3},
        {true, 1, 0, 0, 4},
    }};
    for (const auto& mode : modes) {
        constant(sources[0], mode, 1);
        constant(sources[2], mode, 1);
    }
    constant({0.9f, 0.1f, 0.6f, 0.3f}, AlphaBlend, Overlaps);
    for (const auto& mode : {AlphaBlend, ColorTargetBlend{true, 1, 5, 0, 7}, ColorTargetBlend{true, 9, 2, 4, 6}}) {
        const auto before = Fill(block, seed++);
        Draw(device, color, mode, Float32Export, VaryingProgram());
        ++draws;
        Expect(device, block, layout, before, mode, 1, VaryingColor, "a blend of a color from the pixel position with factors " + std::to_string(mode.sourceFactor) + "/" + std::to_string(mode.destinationFactor) + ", function " + std::to_string(mode.function));
    }
    const std::array<float, 4> halves{0.5f, 0.75f, 0.125f, 0.375f};
    const auto before = Fill(block, seed++);
    Draw(device, color, AlphaBlend, Float16Export, HalfProgram(halves));
    ++draws;
    Expect(device, block, layout, before, AlphaBlend, 1, [&](std::uint32_t, std::uint32_t) { return halves; }, "an FP16_ABGR blend of " + Describe(halves, AlphaBlend));
    for (const bool exportFirst : {false, true}) {
        const auto killedBefore = Fill(block, seed++);
        Draw(device, color, AlphaBlend, Float32Export, KillingProgram(sources[0], exportFirst));
        ++draws;
        ExpectWhere(device, block, layout, killedBefore, AlphaBlend, 1, [&](std::uint32_t, std::uint32_t) { return sources[0]; }, [](std::uint32_t x, std::uint32_t y) { return y < x; }, std::string(exportFirst ? "a blend from a pixel shader that clears EXEC between its color export and exp null done vm" : "a blend from a pixel shader that clears EXEC before exp mrt0 done vm") + " of " + Describe(sources[0], AlphaBlend));
    }
    return draws;
}

void MixedTest(AgcDriver::VulkanDevice& device, Block& block) {
    const ColorTargetLayout layout(Width, Height, ColorTileMode::RenderTarget, 4);
    const auto color = DecodeTarget(block, TiledAttrib3);
    auto& attached = Attached;
    std::fill(attached.pixels.begin(), attached.pixels.end(), std::uint8_t{0x5a});
    const std::array<float, 4> blendedColor{0.6f, 0.2f, 0.9f, 0.25f};
    const std::array<float, 4> attachedColor{1.0f, 0.0f, 64.0f / 255.0f, 200.0f / 255.0f};
    const auto before = Fill(block, 99);
    Draw(device, color, AlphaBlend, Float32Export, MixedProgram(blendedColor, attachedColor), 1, &attached);
    device.WaitIdle();
    Expect(device, block, layout, before, AlphaBlend, 1, [&](std::uint32_t, std::uint32_t) { return blendedColor; }, "a blend beside an RGBA8 attachment of " + Describe(blendedColor, AlphaBlend));
    const std::array<std::uint8_t, 4> written{255, 0, 64, 200};
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            for (std::uint32_t channel = 0; channel < 4; ++channel) {
                const auto expected = x < ScissorWidth ? written[channel] : std::uint8_t{0x5a};
                const auto value = attached.pixels[(static_cast<std::size_t>(y) * Width + x) * 4u + channel];
                if (value == expected) continue;
                throw std::runtime_error("the RGBA8 attachment beside a blended 10_11_11 unorm target holds " + std::to_string(value) + " at (" + std::to_string(x) + ", " + std::to_string(y) + ") channel " + std::to_string(channel) + ", expected " + std::to_string(expected));
            }
        }
    }
}

void LinearTest(AgcDriver::VulkanDevice& device, Block& block) {
    const auto color = DecodeTarget(block, LinearAttrib3);
    std::string failure;
    try {
        Draw(device, color, AlphaBlend, Float32Export, ConstantProgram({0.25f, 0.5f, 0.75f, 0.5f}));
    } catch (const std::exception& error) {
        failure = error.what();
    }
    device.WaitIdle();
    Require(failure.find("blending into a linear 10_11_11 unorm color target is unsupported") != std::string::npos, "a blend into a linear 10_11_11 unorm target reported \"" + failure + "\"");
}

void UnclampedByteTests(AgcDriver::VulkanDevice& device, std::uint32_t channels) {
    const ColorTargetLayout layout(Width, Height, ColorTileMode::RenderTarget, channels);
    Block block(layout.Bytes());
    AgcDriver::Registers cx;
    cx[0x318] = static_cast<std::uint32_t>(block.Address() >> 8u);
    cx[0x31b] = 0;
    cx[0x31c] = channels == 1u ? 0x4u : channels == 2u ? 0xcu : 0x28u;
    cx[0x31d] = 0;
    cx[0x3b0] = ((Width - 1u) << 14u) | (Height - 1u);
    cx[0x3b8] = TiledAttrib3;
    cx[0x390] = static_cast<std::uint32_t>(block.Address() >> 40u);
    auto color = AgcDriver::Graphics::DecodeColorBuffer(cx, 0);
    color.format = channels == 1u ? VK_FORMAT_R8_UINT : channels == 2u ? VK_FORMAT_R16_UINT : VK_FORMAT_R32_UINT;
    color.packing = channels == 1u ? ShaderRecompiler::ColorExportPacking::UnclampedUnorm8 : channels == 2u ? ShaderRecompiler::ColorExportPacking::UnclampedUnorm8_8 : ShaderRecompiler::ColorExportPacking::UnclampedUnorm8_8_8_8;
    const std::array<std::array<float, 4>, 4> sources{{
        {-0.5f, 1.5f, 0.0f, 0.25f}, {2.0f, -1.0f, 0.0f, 0.5f},
        {0.25f, 0.75f, 0.0f, 1.5f}, {0.25f, 0.75f, 0.0f, -0.5f}
    }};
    std::size_t draws = 0;
    const auto masks = channels == 1u ? std::vector<std::uint32_t>{1u} : channels == 2u ? std::vector<std::uint32_t>{1u, 2u, 3u} : std::vector<std::uint32_t>{1u, 2u, 4u, 8u, 15u};
    for (const auto& source : sources) {
        for (const auto exportFormat : {Float32Export, Float16Export}) {
            for (const auto mask : masks) {
                for (const auto operation : {0u, 1u, 4u}) {
                    const ColorTargetBlend blend{true, 4u, 5u, static_cast<std::uint8_t>(operation), static_cast<std::uint8_t>(mask)};
                    const auto before = Fill(block, static_cast<std::uint32_t>(draws + 1u));
                    Draw(device, color, blend, exportFormat, exportFormat == Float32Export ? ConstantProgram(source) : HalfProgram(source), Overlaps);
                    const auto stored = ReadBack(device, block);
                    for (std::uint32_t y = 0; y < Height; ++y) {
                        for (std::uint32_t x = 0; x < Width; ++x) {
                            const auto offset = layout.Offset(x, y);
                            std::uint32_t expected = 0, actual = 0;
                            std::memcpy(&expected, reinterpret_cast<const std::byte*>(before.data()) + offset, channels);
                            std::memcpy(&actual, reinterpret_cast<const std::byte*>(stored.data()) + offset, channels);
                            if (x < ScissorWidth) {
                                for (std::uint32_t step = 0; step < Overlaps; ++step) {
                                    for (std::uint32_t lane = 0; lane < channels; ++lane) {
                                        if ((mask & (1u << lane)) == 0u) continue;
                                        const auto shift = lane * 8u;
                                        const float destination = static_cast<float>((expected >> shift) & 255u) / 255.0f;
                                        volatile float sourceTerm = source[lane] * source[3];
                                        volatile float destinationTerm = destination * (1.0f - source[3]);
                                        const auto value = operation == 0u ? sourceTerm + destinationTerm : operation == 1u ? sourceTerm - destinationTerm : destinationTerm - sourceTerm;
                                        const auto field = Unorm(value, 8u);
                                        expected = (expected & ~(255u << shift)) | (field << shift);
                                    }
                                }
                            }
                            Require(actual == expected, "unclamped " + std::to_string(channels) + "-channel blend mismatch at " + std::to_string(x) + "," + std::to_string(y) + ": " + std::to_string(actual) + " instead of " + std::to_string(expected));
                        }
                    }
                    ++draws;
                }
            }
        }
    }
    const ColorTargetLayout wideLayout(Width, Height, ColorTileMode::RenderTarget, 4);
    Block wide(wideLayout.Bytes());
    const auto wideColor = DecodeTarget(wide, TiledAttrib3);
    const std::array<float, 4> bounded{0.25f, 0.75f, 0.5f, 0.5f};
    const auto program = ConstantProgram(bounded);
    const auto before = Fill(wide, 113u);
    Draw(device, wideColor, AlphaBlend, Float32Export, program);
    Draw(device, color, ColorTargetBlend{true, 4u, 5u, 0u, static_cast<std::uint8_t>((1u << channels) - 1u)}, Float32Export, program);
    Draw(device, wideColor, AlphaBlend, Float32Export, program);
    Expect(device, wide, wideLayout, before, AlphaBlend, 2u, [&](std::uint32_t, std::uint32_t) { return bounded; }, "an R32ui blend reused after an R16ui specialization of the same shader");
    const auto lastBefore = Fill(block, 157u);
    const std::array<float, 4> lastColor{1.0f, 0.0f, 1.0f, 1.0f};
    Draw(device, color, ColorTargetBlend{true, 4u, 5u, 0u, static_cast<std::uint8_t>((1u << channels) - 1u)}, Float32Export, ConstantProgram(lastColor, 7u), 1u, nullptr, 7u);
    const auto lastStored = ReadBack(device, block);
    const std::array<std::uint8_t, 4> expectedBytes{255u, 0u, 255u, 255u};
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto offset = layout.Offset(x, y);
            for (std::uint32_t lane = 0; lane < channels; ++lane) {
                const auto expected = x < ScissorWidth ? expectedBytes[lane] : reinterpret_cast<const std::uint8_t*>(lastBefore.data())[offset + lane];
                Require(reinterpret_cast<const std::uint8_t*>(lastStored.data())[offset + lane] == expected, "an unclamped blend through MRT7 wrote the wrong texel");
            }
        }
    }
    Block second(layout.Bytes());
    auto secondColor = color;
    secondColor.address = second.Address();
    secondColor.slot = 7u;
    secondColor.exportIndex = 7u;
    const auto firstBefore = Fill(block, 171u);
    const auto secondBefore = Fill(second, 183u);
    Draw(device, color, ColorTargetBlend{true, 4u, 5u, 0u, static_cast<std::uint8_t>((1u << channels) - 1u)}, Float32Export, ConstantProgram(lastColor, 0u, 7u), 1u, nullptr, 0u, &secondColor);
    for (const auto& [target, initial] : std::array<std::pair<const Block*, const std::vector<std::uint32_t>*>, 2>{{{&block, &firstBefore}, {&second, &secondBefore}}}) {
        const auto stored = ReadBack(device, *target);
        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                const auto offset = layout.Offset(x, y);
                for (std::uint32_t lane = 0; lane < channels; ++lane) {
                    const auto expected = x < ScissorWidth ? expectedBytes[lane] : reinterpret_cast<const std::uint8_t*>(initial->data())[offset + lane];
                    Require(reinterpret_cast<const std::uint8_t*>(stored.data())[offset + lane] == expected, "simultaneous unclamped blends wrote the wrong texel");
                }
            }
        }
    }
    std::printf("Unclamped %u-channel UNORM blending passed: %zu draws, FP32/FP16 exports, out-of-range colors and alpha, %u overlapping fragments, channel masks, scissor, MRT7, simultaneous targets and shader reuse\n", channels, draws + 5u, Overlaps);
}

}

int main(int argc, char** argv) {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto capabilities = device->Target().supportedCapabilities;
        if (std::find(capabilities.begin(), capabilities.end(), PixelInterlockCapability) == capabilities.end()) {
            std::puts("skipped, the device lacks fragmentShaderPixelInterlock");
            return VulkanTestSkipped;
        }
        if (argc == 2 && std::string(argv[1]) == "--unclamped-pair") {
            for (const auto channels : {1u, 2u, 4u}) UnclampedByteTests(*device, channels);
            return 0;
        }
        Block tiled(ColorTargetLayout(Width, Height, ColorTileMode::RenderTarget, 4).Bytes());
        Block linear(ColorTargetLayout(Width, Height, ColorTileMode::Linear, 4).Bytes());
        const auto draws = TargetTests(*device, tiled);
        MixedTest(*device, tiled);
        LinearTest(*device, linear);
        std::printf("10_11_11 unorm blending tests passed: %zu draws compared bit for bit with the fp32 reference, two killing pixels and one beside an RGBA8 attachment\n", draws);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
