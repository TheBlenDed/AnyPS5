#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

constexpr uint64_t LOWER_ADDRESS_SIZE = uint64_t{1} << 40;
namespace LibKernel::Memory {
constexpr uint64_t kExtendedMemoryBase = uint64_t{1} << 45;
constexpr uint64_t kExtendedMemorySize = uint64_t{1} << 38;
}
struct Buffer {
    uint64_t begin, size;
    int score;
    uint64_t CpuAddress() const { return begin; }
    uint64_t Size() const { return size; }
    int StreamScore() const { return score; }
};
struct BufferCache {
    static constexpr uint64_t CACHING_PAGESIZE = 16384;
    std::map<uint64_t, size_t> m_buffers;
    std::vector<Buffer> m_slot_buffers;
    struct OverlapResult {
        decltype(m_buffers)::iterator first, last;
        uint64_t begin, end;
        bool leap;
    };
    OverlapResult ResolveOverlaps(uint64_t, uint64_t);
};
