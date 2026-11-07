#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include "prx/libSceAgcDriver/Execution/include/StackWalk.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/CommitterOverlap.hpp"
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stop_token>
#include <system_error>
#include <thread>
#include <vector>
#endif

namespace AgcDriver {

#ifdef _WIN32
namespace {

std::jthread processSamplerThread;
std::jthread workerSamplerThread;
std::jthread committerSamplerThread;
std::jthread committerOverlapSamplerThread;

// Debug aid: APS5_SAMPLE_WORKER=<file> samples the calling thread every millisecond and, every 20 s,
// writes "module+offset self inclusive" lines to <file> for offline symbolization with nm.
struct Sampler {
    static constexpr int ProcessDepth = 12;
    static constexpr int WorkerDepth = 64;

    HANDLE target = nullptr;
    std::string path;
    int depth = ProcessDepth;
    bool keepStacks = false;
    std::uint64_t stackLow = 0;
    std::uint64_t stackHigh = 0;
    std::vector<DwarfFrames> modules;
    std::uint64_t pendingModule = 0;
    AgcDriver::Mutex mutex;
    std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> counts;
    CollapsedStacks stacks;
    std::uint64_t samples = 0;
    std::int64_t suspendedTicks = 0;
    std::int64_t longestSuspendedTicks = 0;

    ~Sampler() noexcept(false) {
        if (target != nullptr && !CloseHandle(target)) throw std::system_error(GetLastError(), std::system_category(), "Closing sampler thread handle");
    }

    bool readStack(std::uint64_t address, std::uint64_t& value) const {
        if (address < stackLow || address > stackHigh - sizeof(value)) return false;
        std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
        return true;
    }

    const DwarfFrames* moduleOf(std::uint64_t pc) const {
        const auto found = std::upper_bound(modules.begin(), modules.end(), pc, [](std::uint64_t value, const DwarfFrames& module) { return value < module.Base(); });
        return found != modules.begin() && std::prev(found)->Contains(pc) ? &*std::prev(found) : nullptr;
    }

    bool hasUnwindData(std::uint64_t pc) const {
        DWORD64 imageBase = 0;
        const auto* module = moduleOf(pc - 1);
        return RtlLookupFunctionEntry(pc, &imageBase, nullptr) != nullptr || (module != nullptr && module->Covers(pc - 1));
    }

    template <typename Read>
    DwarfFrames::Step stepDwarf(CONTEXT& context, bool leaf, const Read& read) {
        const auto* module = moduleOf(context.Rip - (leaf ? 0 : 1));
        if (module == nullptr) {
            MEMORY_BASIC_INFORMATION memory{};
            if (VirtualQuery(reinterpret_cast<LPCVOID>(context.Rip), &memory, sizeof(memory)) != 0 && memory.Type == MEM_IMAGE) pendingModule = reinterpret_cast<std::uint64_t>(memory.AllocationBase);
            return DwarfFrames::Step::Failed;
        }
        DwarfFrames::Registers registers{context.Rax, context.Rdx, context.Rcx, context.Rbx, context.Rsi, context.Rdi, context.Rbp, context.Rsp, context.R8, context.R9, context.R10, context.R11, context.R12, context.R13, context.R14, context.R15, context.Rip};
        const auto step = module->Unwind(registers, leaf, [&](std::uint64_t address, std::uint64_t& value) { return read(address, value); });
        if (step != DwarfFrames::Step::Stepped) return step;
        DWORD64* columns[] = {&context.Rax, &context.Rdx, &context.Rcx, &context.Rbx, &context.Rsi, &context.Rdi, &context.Rbp, &context.Rsp, &context.R8, &context.R9, &context.R10, &context.R11, &context.R12, &context.R13, &context.R14, &context.R15, &context.Rip};
        for (std::size_t column = 0; column < registers.size(); ++column) *columns[column] = registers[column];
        return step;
    }

    int walk(CONTEXT& context, std::array<std::uint64_t, WorkerDepth>& frames) {
        const bool dwarf = stackHigh != 0 && context.Rsp >= stackLow && context.Rsp < stackHigh;
        int count = 0;
        while (count < depth && context.Rip != 0) {
            frames[count++] = context.Rip;
            DWORD64 imageBase = 0;
            if (auto* entry = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr)) {
                PVOID handler = nullptr;
                DWORD64 establisher = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, entry, &context, &handler, &establisher, nullptr);
                continue;
            }
            if (!dwarf) break;
            const auto step = stepDwarf(context, count == 1, [this](std::uint64_t address, std::uint64_t& value) { return readStack(address, value); });
            if (step == DwarfFrames::Step::Stepped) continue;
            std::uint64_t returnAddress = 0;
            if (step == DwarfFrames::Step::Failed || count != 1 || !readStack(context.Rsp, returnAddress) || !hasUnwindData(returnAddress)) break;
            context.Rip = returnAddress;
            context.Rsp += sizeof(returnAddress);
        }
        return count;
    }

    void indexModule() {
        const auto base = std::exchange(pendingModule, 0);
        if (moduleOf(base) != nullptr) return;
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        const auto* sections = IMAGE_FIRST_SECTION(nt);
        const DwarfFrames::Byte* frames = nullptr;
        const DwarfFrames::Byte* framesEnd = nullptr;
        for (unsigned index = 0; index < nt->FileHeader.NumberOfSections; ++index) {
            if (std::memcmp(sections[index].Name, ".ehfram", 8) != 0) continue;
            frames = reinterpret_cast<const DwarfFrames::Byte*>(base + sections[index].VirtualAddress);
            framesEnd = frames + sections[index].Misc.VirtualSize;
        }
        const auto at = std::upper_bound(modules.begin(), modules.end(), base, [](std::uint64_t value, const DwarfFrames& module) { return value < module.Base(); });
        modules.emplace(at, base, base + nt->OptionalHeader.SizeOfImage, frames, framesEnd);
    }

    void sample() {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        std::array<std::uint64_t, WorkerDepth> frames;
        int count = 0;
        LARGE_INTEGER suspended{};
        LARGE_INTEGER resumed{};
        QueryPerformanceCounter(&suspended);
        if (SuspendThread(target) == static_cast<DWORD>(-1)) return;
        if (GetThreadContext(target, &context) != 0) count = walk(context, frames);
        ResumeThread(target);
        QueryPerformanceCounter(&resumed);
        if (pendingModule != 0) indexModule();
        std::lock_guard lock(mutex);
        ++samples;
        suspendedTicks += resumed.QuadPart - suspended.QuadPart;
        longestSuspendedTicks = std::max(longestSuspendedTicks, resumed.QuadPart - suspended.QuadPart);
        for (int i = 0; i < count; ++i) {
            auto& entry = counts[frames[i]];
            if (i == 0) ++entry.first;
            ++entry.second;
        }
        if (keepStacks && count != 0) stacks.Add({frames.data(), static_cast<std::size_t>(count)}, context.Rip != 0);
    }

    static std::string frameName(std::uint64_t address, std::map<std::uint64_t, std::string>& loaded) {
        HMODULE module = nullptr;
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module)) GetModuleFileNameA(module, name, sizeof(name));
        const char* base = std::strrchr(name, '\\');
        if (module != nullptr) loaded.emplace(reinterpret_cast<std::uint64_t>(module), base ? base + 1 : name);
        char frame[MAX_PATH + 32];
        std::snprintf(frame, sizeof(frame), "%s+0x%llx", base ? base + 1 : name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)));
        return frame;
    }

    static void writeModules(std::FILE* file, const std::map<std::uint64_t, std::string>& loaded) {
        for (const auto& [base, name] : loaded) {
            const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
            const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
            std::fprintf(file, "# module %s 0x%lx 0x%lx\n", name.c_str(), static_cast<unsigned long>(nt->OptionalHeader.SizeOfImage), static_cast<unsigned long>(nt->FileHeader.TimeDateStamp));
        }
    }

    void write() {
        std::lock_guard lock(mutex);
        std::FILE* file = std::fopen(path.c_str(), "w");
        if (file == nullptr) return;
        std::fprintf(file, "# %llu samples\n", static_cast<unsigned long long>(samples));
        std::map<std::uint64_t, std::string> loaded;
        for (const auto& [address, count] : counts) {
            HMODULE module = nullptr;
            char name[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module)) GetModuleFileNameA(module, name, sizeof(name));
            const char* base = std::strrchr(name, '\\');
            if (module != nullptr) loaded.emplace(reinterpret_cast<std::uint64_t>(module), base ? base + 1 : name);
            std::fprintf(file, "%s 0x%llx %llu %llu\n", base ? base + 1 : name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)), static_cast<unsigned long long>(count.first), static_cast<unsigned long long>(count.second));
        }
        writeModules(file, loaded);
        std::fclose(file);
        if (keepStacks) writeStacks();
    }

    void writeStacks() {
        const auto dot = path.rfind('.');
        const auto slash = path.find_last_of("\\/");
        const auto stem = dot != std::string::npos && (slash == std::string::npos || dot > slash) ? path.substr(0, dot) : path;
        const auto collapsed = stem + ".collapsed.txt";
        const auto temporary = collapsed + ".tmp";
        std::map<std::uint64_t, std::string> names;
        std::map<std::uint64_t, std::string> loaded;
        for (const auto& [address, count] : counts) names.emplace(address, frameName(address, loaded));
        std::FILE* file = std::fopen(temporary.c_str(), "w");
        if (file == nullptr) return;
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        const double tickUs = 1e6 / static_cast<double>(frequency.QuadPart);
        std::fprintf(file, "# samples=%llu depth=%d truncated=%llu suspended_us_mean=%.1f suspended_us_max=%.1f\n", static_cast<unsigned long long>(samples), depth, static_cast<unsigned long long>(stacks.Truncated()),
            samples != 0 ? static_cast<double>(suspendedTicks) * tickUs / static_cast<double>(samples) : 0.0, static_cast<double>(longestSuspendedTicks) * tickUs);
        writeModules(file, loaded);
        const auto text = stacks.Format([&](std::uint64_t address) { return names.at(address); });
        const bool written = std::fwrite(text.data(), 1, text.size(), file) == text.size();
        if (std::fclose(file) != 0 || !written) return;
        MoveFileExA(temporary.c_str(), collapsed.c_str(), MOVEFILE_REPLACE_EXISTING);
    }
};

// Debug aid: APS5_SAMPLE_THREADS=<file> samples every thread in the process every 2 ms and, every
// 20 s, writes each busy thread's hottest frames ("module+offset self inclusive") to <file>.
struct ProcessSampler {
    std::string path;
    DWORD self = 0;
    std::map<DWORD, Sampler> threads;
    std::uint64_t rounds = 0;

    std::chrono::steady_clock::time_point enumerated{};

    // The system-wide thread snapshot is slow (tens of ms); the thread list is refreshed every 5 s and
    // the rounds in between only sample the known threads.
    void enumerate() {
        const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return;
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        const DWORD process = GetCurrentProcessId();
        for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) continue;
            auto& thread = threads[entry.th32ThreadID];
            if (thread.target == nullptr) thread.target = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
        }
        CloseHandle(snapshot);
        enumerated = std::chrono::steady_clock::now();
    }

    void sample() {
        if (threads.empty() || std::chrono::steady_clock::now() - enumerated > std::chrono::seconds(5)) enumerate();
        for (auto& [id, thread] : threads) {
            if (thread.target != nullptr) thread.sample();
        }
        ++rounds;
    }

    void write() {
        std::FILE* file = std::fopen(path.c_str(), "w");
        if (file == nullptr) return;
        std::fprintf(file, "# %llu rounds\n", static_cast<unsigned long long>(rounds));
        std::map<std::uint64_t, std::string> loaded;
        for (auto& [id, thread] : threads) {
            std::lock_guard lock(thread.mutex);
            std::vector<std::pair<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>>> hot(thread.counts.begin(), thread.counts.end());
            std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
            std::uint64_t leaves = 0;
            for (const auto& [address, count] : hot) leaves += count.first;
            if (leaves == 0) continue;
            std::fprintf(file, "thread %lu samples %llu\n", static_cast<unsigned long>(id), static_cast<unsigned long long>(thread.samples));
            for (std::size_t i = 0; i < hot.size() && i < 40; ++i) {
                const auto address = hot[i].first;
                HMODULE module = nullptr;
                char name[MAX_PATH] = "?";
                if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module)) GetModuleFileNameA(module, name, sizeof(name));
                const char* base = std::strrchr(name, '\\');
                if (module != nullptr) loaded.emplace(reinterpret_cast<std::uint64_t>(module), base ? base + 1 : name);
                std::fprintf(file, "  %s 0x%llx %llu %llu\n", base ? base + 1 : name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)), static_cast<unsigned long long>(hot[i].second.first), static_cast<unsigned long long>(hot[i].second.second));
            }
        }
        Sampler::writeModules(file, loaded);
        std::fclose(file);
    }
};

struct ResumeGuard {
    HANDLE thread = nullptr;
    bool active = false;
    explicit ResumeGuard(HANDLE handle) : thread(handle), active(true) {}
    ~ResumeGuard() {
        if (active) ResumeThread(thread);
    }
    void dismiss() { active = false; }
};

struct CapturedStack {
    std::uint64_t base;
    std::span<const std::byte> bytes;

    bool Read(std::uint64_t address, void* destination, std::size_t size) const {
        if (address < base || size > bytes.size() || address - base > bytes.size() - size) return false;
        std::memcpy(destination, bytes.data() + (address - base), size);
        return true;
    }
};

thread_local const CapturedStack* activeStack = nullptr;

BOOL CALLBACK readCapturedStack(HANDLE process, DWORD64 address, PVOID destination, DWORD size, LPDWORD read) {
    *read = 0;
    if (activeStack->Read(address, destination, size)) {
        *read = size;
        return TRUE;
    }
    MEMORY_BASIC_INFORMATION memory{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) == 0 || memory.Type != MEM_IMAGE || memory.State != MEM_COMMIT) return FALSE;
    const auto region = reinterpret_cast<std::uint64_t>(memory.BaseAddress);
    if (size > memory.RegionSize || address - region > memory.RegionSize - size) return FALSE;
    SIZE_T copied = 0;
    const BOOL ok = ReadProcessMemory(process, reinterpret_cast<const void*>(address), destination, size, &copied);
    *read = static_cast<DWORD>(copied);
    return ok;
}

PVOID CALLBACK sampledFunctionEntry(HANDLE, DWORD64 pc) {
    DWORD64 base = 0;
    return RtlLookupFunctionEntry(pc, &base, nullptr);
}

DWORD64 CALLBACK sampledModuleBase(HANDLE, DWORD64 pc) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(pc), &module)) return 0;
    return reinterpret_cast<DWORD64>(module);
}

decltype(&StackWalk64) capturedStackWalker() {
    static const auto function = [] {
        const auto module = LoadLibraryExW(L"dbghelp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        return module != nullptr ? reinterpret_cast<decltype(&StackWalk64)>(GetProcAddress(module, "StackWalk64")) : nullptr;
    }();
    return function;
}

struct OverlapSampler {
    static constexpr int PeriodMs = 5;
    static constexpr std::size_t StackBytes = 16 * 1024;
    static constexpr std::size_t MaxFrames = 24;
    static constexpr std::int64_t BucketNs = 500'000'000;
    static constexpr std::uint64_t CaptureCap = 200'000;

    struct Bucket {
        CollapsedStacks stacks;
        std::uint64_t samples = 0;
    };

    Sampler sampler;
    std::string path;
    std::array<std::byte, StackBytes> stack;
    std::map<std::uint64_t, Bucket> buckets;
    std::map<std::uint64_t, std::uint64_t> counts;
    std::uint64_t polls = 0;
    std::uint64_t covered = 0;
    std::uint64_t missed = 0;
    std::uint64_t captured = 0;
    std::uint64_t dropped = 0;
    std::uint64_t rejected = 0;
    std::uint64_t failed = 0;

    void tick() {
        const auto gate = DriverDetail::CommitterOverlap::Get().OverlapGateValue();
        ++polls;
        if ((gate & 1u) == 0u) {
            ++missed;
            return;
        }
        ++covered;
        if (captured >= CaptureCap) {
            ++dropped;
            return;
        }
        const auto time = std::chrono::steady_clock::now();
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        SIZE_T copied = 0;
        if (SuspendThread(sampler.target) == static_cast<DWORD>(-1)) {
            ++failed;
            return;
        }
        ResumeGuard guard{sampler.target};
        const auto capturedGate = DriverDetail::CommitterOverlap::Get().OverlapGateValue();
        const bool haveContext = capturedGate == gate && GetThreadContext(sampler.target, &context) != 0;
        const auto base = context.Rsp;
        if (haveContext && base >= sampler.stackLow && base < sampler.stackHigh) {
            const auto bytes = std::min<std::uint64_t>(stack.size(), sampler.stackHigh - base);
            if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(base), stack.data(), bytes, &copied)) copied = 0;
        }
        const auto afterCapture = DriverDetail::CommitterOverlap::Get().OverlapGateValue();
        const DWORD resumed = ResumeThread(sampler.target);
        if (resumed == static_cast<DWORD>(-1)) {
            ++failed;
            return;
        }
        guard.dismiss();
        if (capturedGate != gate || afterCapture != gate) {
            ++rejected;
            return;
        }
        if (!haveContext || context.Rip == 0) {
            ++failed;
            return;
        }
        const CapturedStack snapshot{base, {stack.data(), copied}};
        std::array<std::uint64_t, MaxFrames> frames;
        frames[0] = context.Rip;
        std::size_t count = 1;
        bool truncated = true;
        const auto nativeWalk = capturedStackWalker();
        if (copied != 0 && nativeWalk != nullptr) {
            CONTEXT native = context;
            STACKFRAME64 frame{};
            frame.AddrPC = {native.Rip, 0, AddrModeFlat};
            frame.AddrFrame = {native.Rbp, 0, AddrModeFlat};
            frame.AddrStack = {native.Rsp, 0, AddrModeFlat};
            activeStack = &snapshot;
            for (std::size_t i = 0; i < MaxFrames && count < frames.size(); ++i) {
                if (!nativeWalk(IMAGE_FILE_MACHINE_AMD64, GetCurrentProcess(), sampler.target, &frame, &native, readCapturedStack, sampledFunctionEntry, sampledModuleBase, nullptr)) break;
                if (frame.AddrPC.Offset == 0) {
                    truncated = false;
                    break;
                }
                if (i != 0 || frame.AddrPC.Offset != frames[0]) frames[count++] = frame.AddrPC.Offset;
                if (frame.AddrReturn.Offset == 0) {
                    truncated = false;
                    break;
                }
            }
            activeStack = nullptr;
        }
        if (copied != 0 && count == 1 && truncated) {
            const auto read = [&](std::uint64_t address, std::uint64_t& value) {
                return snapshot.Read(address, &value, sizeof(value));
            };
            while (count < frames.size()) {
                auto step = sampler.stepDwarf(context, count == 1, read);
                if (sampler.pendingModule != 0) {
                    sampler.indexModule();
                    step = sampler.stepDwarf(context, count == 1, read);
                }
                if (step != DwarfFrames::Step::Stepped) break;
                if (context.Rip == 0) {
                    truncated = false;
                    break;
                }
                frames[count++] = context.Rip;
            }
        }
        const auto key = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count() / BucketNs);
        auto& bucket = buckets[key];
        bucket.stacks.Add({frames.data(), count}, truncated);
        ++bucket.samples;
        for (std::size_t i = 0; i < count; ++i) ++counts[frames[i]];
        ++captured;
    }

    void write() {
        std::FILE* file = std::fopen(path.c_str(), "a");
        if (file == nullptr) {
            for (const auto& [key, bucket] : buckets) dropped += bucket.samples;
            buckets.clear();
            counts.clear();
            return;
        }
        std::uint64_t truncated = 0;
        for (const auto& [key, bucket] : buckets) truncated += bucket.stacks.Truncated();
        std::fprintf(file, "# overlap polls=%llu covered=%llu missed=%llu captured=%llu dropped=%llu rejected=%llu failed=%llu cadence_ms=%d bucket_ms=%lld truncated=%llu\n",
            static_cast<unsigned long long>(polls), static_cast<unsigned long long>(covered), static_cast<unsigned long long>(missed),
            static_cast<unsigned long long>(captured), static_cast<unsigned long long>(dropped), static_cast<unsigned long long>(rejected),
            static_cast<unsigned long long>(failed), PeriodMs, static_cast<long long>(BucketNs / 1'000'000), static_cast<unsigned long long>(truncated));
        std::map<std::uint64_t, std::string> names;
        std::map<std::uint64_t, std::string> loaded;
        for (const auto& [address, count] : counts) names.emplace(address, Sampler::frameName(address, loaded));
        Sampler::writeModules(file, loaded);
        for (const auto& [key, bucket] : buckets) {
            std::fprintf(file, "# bucket %llu begin_ns %lld samples %llu\n", static_cast<unsigned long long>(key), static_cast<long long>(key * BucketNs), static_cast<unsigned long long>(bucket.samples));
            const auto text = bucket.stacks.Format([&](std::uint64_t address) { return names.at(address); });
            std::fwrite(text.data(), 1, text.size(), file);
        }
        std::fclose(file);
        buckets.clear();
        counts.clear();
    }
};

void StartProcessSampler() {
    const char* path = std::getenv("APS5_SAMPLE_THREADS");
    if (path == nullptr) return;
    processSamplerThread = std::jthread([path = std::string(path)](std::stop_token token) {
        auto sampler = std::make_unique<ProcessSampler>();
        sampler->path = path;
        sampler->self = GetCurrentThreadId();
        auto flushed = std::chrono::steady_clock::now();
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            sampler->sample();
            if (std::chrono::steady_clock::now() - flushed > std::chrono::seconds(20)) {
                sampler->write();
                flushed = std::chrono::steady_clock::now();
            }
        }
        sampler->write();
    });
}

void StartCallingThreadSampler(const char* path, std::jthread& slot) {
    if (path == nullptr) return;
    auto sampler = std::make_unique<Sampler>();
    sampler->path = path;
    sampler->depth = Sampler::WorkerDepth;
    sampler->keepStacks = true;
    const auto* block = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    MEMORY_BASIC_INFORMATION stack{};
    if (VirtualQuery(block->StackLimit, &stack, sizeof(stack)) != 0) {
        sampler->stackLow = reinterpret_cast<std::uint64_t>(stack.AllocationBase);
        sampler->stackHigh = reinterpret_cast<std::uint64_t>(block->StackBase);
    }
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &sampler->target, THREAD_ALL_ACCESS, FALSE, 0)) throw std::system_error(GetLastError(), std::system_category(), "Duplicating sampler thread handle");
    slot = std::jthread([sampler = std::move(sampler)](std::stop_token token) {
        auto flushed = std::chrono::steady_clock::now();
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            sampler->sample();
            if (std::chrono::steady_clock::now() - flushed > std::chrono::seconds(20)) {
                sampler->write();
                flushed = std::chrono::steady_clock::now();
            }
        }
        sampler->write();
    });
}

}

void StartCommitterOverlapSampler() {
    const char* path = std::getenv("APS5_SAMPLE_OVERLAP");
    if (path == nullptr || *path == '\0' || !DriverDetail::CommitterOverlap::Enabled()) return;
    auto overlap = std::make_unique<OverlapSampler>();
    overlap->path = path;
    const auto* block = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    MEMORY_BASIC_INFORMATION stack{};
    if (VirtualQuery(block->StackLimit, &stack, sizeof(stack)) != 0) {
        overlap->sampler.stackLow = reinterpret_cast<std::uint64_t>(stack.AllocationBase);
        overlap->sampler.stackHigh = reinterpret_cast<std::uint64_t>(block->StackBase);
    }
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &overlap->sampler.target, THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, 0)) return;
    committerOverlapSamplerThread = std::jthread([overlap = std::move(overlap)](std::stop_token token) {
        auto flushed = std::chrono::steady_clock::now();
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(OverlapSampler::PeriodMs));
            overlap->tick();
            if (std::chrono::steady_clock::now() - flushed > std::chrono::seconds(20)) {
                overlap->write();
                flushed = std::chrono::steady_clock::now();
            }
        }
        overlap->write();
    });
}

void StartWorkerSampler() {
    StartProcessSampler();
    StartCallingThreadSampler(std::getenv("APS5_SAMPLE_WORKER"), workerSamplerThread);
}

void StartCommitterSampler() {
    StartCallingThreadSampler(std::getenv("APS5_SAMPLE_COMMITTER"), committerSamplerThread);
}

void StopWorkerSampler() {
    processSamplerThread.request_stop();
    workerSamplerThread.request_stop();
    committerSamplerThread.request_stop();
    committerOverlapSamplerThread.request_stop();
    if (processSamplerThread.joinable()) processSamplerThread.join();
    if (workerSamplerThread.joinable()) workerSamplerThread.join();
    if (committerSamplerThread.joinable()) committerSamplerThread.join();
    if (committerOverlapSamplerThread.joinable()) committerOverlapSamplerThread.join();
}
#else
void StartWorkerSampler() {}
void StartCommitterSampler() {}
void StartCommitterOverlapSampler() {}
void StopWorkerSampler() {}
#endif

}
