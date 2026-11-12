#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver;
using namespace AgcDriver::Graphics;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::uint32_t Written = 42;
constexpr std::uint32_t Signaled = 43;
constexpr std::uint32_t PeerWritten = 44;
constexpr int FailingCleanup = 8;

void SleepBriefly() {
    std::this_thread::sleep_for(std::chrono::microseconds(200));
}

class TimelineGate {
public:
    explicit TimelineGate(VulkanDevice& device) : device(device) {
        const auto& context = device.graphicsContext();
        const auto create = context.Function<PFN_vkCreateSemaphore>("vkCreateSemaphore");
        destroy = context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore");
        signal = context.Function<PFN_vkSignalSemaphoreKHR>("vkSignalSemaphoreKHR");
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        Require(create(context.device, &info, nullptr, &semaphore) == VK_SUCCESS, "cannot create the freshness gate semaphore");
    }

    ~TimelineGate() {
        try {
            Release();
            std::lock_guard guard(GuestMemory::GpuMutex());
            device.WaitIdle();
        } catch (...) {
        }
        destroy(device.graphicsContext().device, semaphore, nullptr);
    }

    TimelineGate(const TimelineGate&) = delete;
    TimelineGate& operator=(const TimelineGate&) = delete;

    void Arm(Recorder& recorder) {
        recorder.Commands();
        recorder.HoldNextSubmit(semaphore, 1);
    }

    VkResult Release() {
        std::lock_guard guard(mutex);
        if (!released) {
            released = true;
            VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            info.semaphore = semaphore;
            info.value = 1;
            result = signal(device.graphicsContext().device, &info);
        }
        return result;
    }

    VkResult Result() {
        std::lock_guard guard(mutex);
        return result;
    }

private:
    VulkanDevice& device;
    PFN_vkDestroySemaphore destroy = nullptr;
    PFN_vkSignalSemaphoreKHR signal = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    std::mutex mutex;
    bool released = false;
    VkResult result = VK_NOT_READY;
};

class Staff {
public:
    void Spawn(std::function<void()> body) {
        threads.emplace_back([this, body = std::move(body)] {
            try {
                body();
            } catch (...) {
                std::lock_guard guard(mutex);
                if (!error) error = std::current_exception();
            }
        });
    }

    void SetCleanup(std::function<void()> action) {
        cleanup = std::move(action);
    }

    void Join() {
        stop.store(true);
        RunCleanup();
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
        std::lock_guard guard(mutex);
        if (error) std::rethrow_exception(error);
    }

    ~Staff() {
        stop.store(true);
        RunCleanup();
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }

    Staff(const Staff&) = delete;
    Staff& operator=(const Staff&) = delete;
    Staff() = default;

    std::atomic<bool> stop{false};

private:
    void RunCleanup() {
        if (!cleanup) return;
        try {
            cleanup();
        } catch (...) {
        }
    }

    std::mutex mutex;
    std::exception_ptr error;
    std::function<void()> cleanup;
    std::vector<std::thread> threads;
};

struct Values {
    std::uint32_t value = 0;
    std::uint32_t label = 0;
    std::uint32_t valueAtLabel = 0;
    std::atomic<std::uint32_t> writeBacks{0};
    std::atomic<std::uint32_t> peerWriteBacks{0};
    std::atomic<std::uint32_t> labels{0};
};

std::shared_ptr<Buffer> MakeBuffer(const Context& context, std::size_t bytes) {
    return std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

std::uint64_t AddressOf(const void* pointer) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

bool TwoLanes(VulkanDevice& device) {
    auto* recorder = Recorder::Active();
    const auto context = device.graphicsContext();
    return recorder != nullptr && recorder->AsyncLaneActive() && context.queue != context.asyncQueue;
}

int RunLockedControl(VulkanDevice& device) {
    auto& gpu = GuestMemory::GpuMutex();
    std::unique_lock lock(gpu);
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the locked control has no active recorder");
    if (!TwoLanes(device)) {
        std::puts("skipped: the freshness control needs two independent Vulkan lanes");
        return VulkanTestSkipped;
    }
    const auto context = device.graphicsContext();
    recorder->Sync();
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    const std::size_t bytes = sizeof(std::uint32_t);
    auto values = std::make_shared<Values>();
    auto staging = MakeBuffer(context, bytes);
    std::memset(staging->Bytes().data(), 0, bytes);

    {
        const Recorder::LaneScope lane(1);
        fill(recorder->Commands(), staging->Handle(), 0, bytes, Written);
        recorder->OweHostRead();
        recorder->NotePendingWrite(AddressOf(&values->value), bytes);
        recorder->OnComplete([values, staging] {
            std::memcpy(&values->value, staging->Bytes().data(), sizeof(values->value));
            values->writeBacks.fetch_add(1);
        });
    }
    recorder->SubmitAllLanes();
    const auto pending = recorder->DescribePendingWrite(AddressOf(&values->value), bytes);
    Require(pending.has_value() && !pending->open, "the locked control writer was not submitted");

    recorder->SyncThrough(AddressOf(&values->value), bytes);
    Require(values->value == Written && values->writeBacks.load() == 1, "the locked sync did not run the write-back exactly once");
    Require(!recorder->DescribePendingWrite(AddressOf(&values->value), bytes).has_value(), "the locked sync left the writer in flight");
    std::puts("locked control: write-back applied under the mutex, proof complete");
    return 0;
}

int RunEdgeOrder(VulkanDevice& device) {
    auto& gpu = GuestMemory::GpuMutex();
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the edge-order arm has no active recorder");
    if (!TwoLanes(device)) {
        std::puts("skipped: the edge-order arm needs two independent Vulkan lanes");
        return VulkanTestSkipped;
    }
    const auto context = device.graphicsContext();
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    const std::size_t bytes = sizeof(std::uint32_t);
    auto values = std::make_shared<Values>();
    auto staging = MakeBuffer(context, bytes);
    std::memset(staging->Bytes().data(), 0, bytes);
    TimelineGate gate(device);
    Staff staff;
    staff.SetCleanup([&gate] { gate.Release(); });
    std::unique_lock lock(gpu);

    recorder->Sync();
    gate.Arm(*recorder);
    {
        const Recorder::LaneScope lane(1);
        fill(recorder->Commands(), staging->Handle(), 0, bytes, Written);
        recorder->OweHostRead();
        recorder->NotePendingWrite(AddressOf(&values->value), bytes);
        recorder->OnComplete([values, staging] {
            std::memcpy(&values->value, staging->Bytes().data(), sizeof(values->value));
            values->writeBacks.fetch_add(1);
        });
    }
    recorder->SubmitAllLanes();
    const auto gated = recorder->DescribePendingWrite(AddressOf(&values->value), bytes);
    Require(gated.has_value() && !gated->open && !gated->signaled, "the gated producer was not submitted and unsignaled");
    const auto producerSerial = gated->serial;

    {
        const Recorder::LaneScope lane(0);
        recorder->AfterCompletions(AddressOf(&values->label), std::as_bytes(std::span(&Signaled, 1)), 1, 0x20u, false);
        recorder->OnComplete([values] {
            values->valueAtLabel = values->value;
            values->labels.fetch_add(1);
        });
    }
    recorder->SubmitAllLanes();

    staff.Spawn([&] {
        while (!staff.stop.load()) {
            {
                std::lock_guard peerLock(gpu);
                recorder->Reap();
            }
            SleepBriefly();
        }
    });
    staff.Spawn([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        gate.Release();
    });

    lock.unlock();
    recorder->WaitSerial(producerSerial);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (values->labels.load() == 0 && std::chrono::steady_clock::now() < deadline) SleepBriefly();
    staff.Join();
    lock.lock();
    Require(gate.Result() == VK_SUCCESS, "the edge-order gate was not released");
    Require(values->label == Signaled && values->labels.load() == 1, "the later label did not publish");
    Require(values->value == Written && values->writeBacks.load() == 1, "the earlier write-back did not land");
    Require(values->valueAtLabel == Written, "a later label became visible before the earlier write-back it waits for");
    std::printf("edge order: producer serial %llu, label observed write-back %u (the waitOther edge held across an unlocked peer reap)\n", static_cast<unsigned long long>(producerSerial), values->valueAtLabel);
    recorder->Sync();
    return 0;
}

int RunFreshnessUnlock(VulkanDevice& device, bool completeProofRequired) {
    auto& gpu = GuestMemory::GpuMutex();
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the freshness arm has no active recorder");
    if (!TwoLanes(device)) {
        std::puts("skipped: the freshness arm needs two independent Vulkan lanes");
        return VulkanTestSkipped;
    }
    if (!recorder->HasTimeline()) {
        std::puts("skipped: the freshness arm needs timeline semaphores");
        return VulkanTestSkipped;
    }
    std::atomic<bool> peerRecorded{false};
    std::atomic<bool> peerAcquired{false};
    std::atomic<std::uint64_t> peerSerial{0};
    const auto context = device.graphicsContext();
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    const std::size_t bytes = sizeof(std::uint32_t);
    auto values = std::make_shared<Values>();
    auto staging = MakeBuffer(context, bytes);
    std::memset(staging->Bytes().data(), 0, bytes);
    TimelineGate gate(device);
    Staff staff;
    staff.SetCleanup([&gate] { gate.Release(); });
    std::unique_lock lock(gpu);

    recorder->Sync();
    gate.Arm(*recorder);
    {
        const Recorder::LaneScope lane(1);
        fill(recorder->Commands(), staging->Handle(), 0, bytes, Written);
        recorder->OweHostRead();
        recorder->NotePendingWrite(AddressOf(&values->value), bytes);
        recorder->OnComplete([values, staging] {
            std::memcpy(&values->value, staging->Bytes().data(), sizeof(values->value));
            values->writeBacks.fetch_add(1);
        });
    }
    recorder->SubmitAllLanes();
    const auto pending = recorder->DescribePendingWrite(AddressOf(&values->value), bytes);
    Require(pending.has_value() && !pending->open && !pending->signaled, "the gated producer was not submitted and unsignaled");
    const auto producerSerial = pending->serial;

    staff.Spawn([&] {
        std::unique_lock peerLock(gpu);
        peerAcquired.store(true);
        const auto peerStaging = MakeBuffer(context, bytes);
        std::memset(peerStaging->Bytes().data(), 0, bytes);
        {
            const Recorder::LaneScope lane(0);
            fill(recorder->Commands(), peerStaging->Handle(), 0, bytes, PeerWritten);
            recorder->OweHostRead();
            recorder->NotePendingWrite(AddressOf(&values->value), bytes);
            recorder->OnComplete([values, peerStaging] {
                std::memcpy(&values->value, peerStaging->Bytes().data(), sizeof(values->value));
                values->peerWriteBacks.fetch_add(1);
            });
        }
        recorder->SubmitAllLanes();
        const auto peerPending = recorder->DescribePendingWrite(AddressOf(&values->value), bytes);
        peerSerial.store(peerPending.has_value() ? peerPending->serial : 0);
        peerRecorded.store(true);
    });

    staff.Spawn([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!peerRecorded.load() && std::chrono::steady_clock::now() < deadline) SleepBriefly();
        gate.Release();
    });

    recorder->SyncThrough(AddressOf(&values->value), bytes, true, false, completeProofRequired);

    const auto after = recorder->DescribePendingWrite(AddressOf(&values->value), bytes);
    const bool proofComplete = !after.has_value();
    staff.Join();
    Require(gate.Result() == VK_SUCCESS, "the freshness gate was not released");
    Require(peerAcquired.load(), "the peer never acquired the GPU mutex during the wait, so the wait was not unlocked");
    Require(peerSerial.load() != 0 && peerSerial.load() > producerSerial, "the peer did not record a newer writer during the window");
    Require(values->writeBacks.load() == 1, "the earlier writer's completion ran more or less than once");

    const char* arm = completeProofRequired ? "contract" : "hook semantics";
    if (proofComplete) {
        Require(completeProofRequired, "the wait returned a complete proof for a caller that did not request the contract");
        Require(values->peerWriteBacks.load() == 1, "the proof completed without the newer writer's completion");
        Require(values->value == PeerWritten, "the proof completed without the newer writer's bytes");
        std::printf("freshness %s: proof complete after relock, newer writer serial %llu retired, resource value %u\n", arm, static_cast<unsigned long long>(peerSerial.load()), values->value);
        recorder->Sync();
        return 0;
    }

    Require(!completeProofRequired, "the freshness contract returned a stale proof: newer writer serial " + std::to_string(after.has_value() ? after->serial : 0) + " recorded while the mutex was released is still in flight, resource value " + std::to_string(values->value));
    Require(values->peerWriteBacks.load() == 0, "the newer writer's completion ran without its batch retiring");
    Require(after.has_value() && after->serial == peerSerial.load(), "the proof reported a pending writer that is not the peer's");
    Require(values->value == Written, "the newer writer's bytes landed without its completion");
    std::printf("freshness %s: no contract requested, newer writer serial %llu stays in flight (existing hook behaviour, unchanged); resource value %u\n", arm, static_cast<unsigned long long>(after->serial), values->value);
    recorder->Sync();
    return 0;
}

int RunNestedLock(VulkanDevice& device) {
    auto& gpu = GuestMemory::GpuMutex();
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the nested-lock arm has no active recorder");
    if (!TwoLanes(device)) {
        std::puts("skipped: the nested-lock arm needs two independent Vulkan lanes");
        return VulkanTestSkipped;
    }
    if (!recorder->HasTimeline()) {
        std::puts("skipped: the nested-lock arm needs timeline semaphores");
        return VulkanTestSkipped;
    }
    const auto context = device.graphicsContext();
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    const std::size_t bytes = sizeof(std::uint32_t);
    auto values = std::make_shared<Values>();
    auto staging = MakeBuffer(context, bytes);
    std::memset(staging->Bytes().data(), 0, bytes);
    TimelineGate gate(device);
    Staff staff;
    staff.SetCleanup([&gate] { gate.Release(); });

    std::atomic<bool> inCall{false};
    std::atomic<bool> peerAcquiredInside{false};
    staff.Spawn([&] {
        while (!staff.stop.load()) {
            std::unique_lock probe(gpu, std::defer_lock);
            if (!probe.try_lock()) {
                SleepBriefly();
                continue;
            }
            if (inCall.load()) peerAcquiredInside.store(true);
            probe.unlock();
            SleepBriefly();
        }
    });

    {
        std::unique_lock outer(gpu);
        std::unique_lock inner(gpu);
        Require(gpu.DepthOnThisThread() == 2, "the nested arm did not reach a depth of two");
        recorder->Sync();
        gate.Arm(*recorder);
        {
            const Recorder::LaneScope lane(1);
            fill(recorder->Commands(), staging->Handle(), 0, bytes, Written);
            recorder->OweHostRead();
            recorder->NotePendingWrite(AddressOf(&values->value), bytes);
            recorder->OnComplete([values, staging] {
                std::memcpy(&values->value, staging->Bytes().data(), sizeof(values->value));
                values->writeBacks.fetch_add(1);
            });
        }
        recorder->SubmitAllLanes();
        staff.Spawn([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            gate.Release();
        });
        inCall.store(true);
        recorder->SyncThrough(AddressOf(&values->value), bytes, true);
        inCall.store(false);
        Require(values->value == Written && values->writeBacks.load() == 1, "the nested sync did not run the write-back exactly once");
        Require(!peerAcquiredInside.load(), "a depth-two freshness sync released the GPU mutex");
    }
    staff.Join();
    recorder->Sync();
    std::puts("nested lock: a depth-two freshness sync kept the GPU mutex and still completed the writer");
    return 0;
}

int RunWorkerFailure() {
    std::atomic<bool> ran{false};
    {
        Staff staff;
        staff.Spawn([&] {
            ran.store(true);
            throw std::runtime_error("injected worker failure");
        });
        try {
            staff.Join();
        } catch (const std::exception& error) {
            Require(ran.load(), "the failing worker never ran");
            std::printf("failure arm: worker exception joined and reported: %s\n", error.what());
            return 0;
        }
    }
    return FailingCleanup;
}

int RunGatedFailure(VulkanDevice& device) {
    auto& gpu = GuestMemory::GpuMutex();
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the gated failure arm has no active recorder");
    const auto start = std::chrono::steady_clock::now();
    bool injected = false;
    try {
        TimelineGate gate(device);
        Staff staff;
        staff.SetCleanup([&gate] { gate.Release(); });
        std::unique_lock lock(gpu);
        gate.Arm(*recorder);
        staff.Spawn([&gpu] { std::unique_lock peerLock(gpu); });
        throw std::runtime_error("injected failure after the gate was armed and a worker was running");
    } catch (const std::exception& error) {
        injected = true;
        std::printf("gated failure: joined and propagated: %s\n", error.what());
    }
    Require(injected, "the gated failure arm did not observe the injected exception");
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    Require(elapsed < 5000, "the gated failure cleanup took " + std::to_string(elapsed) + " ms, so a worker was not bounded");
    std::puts("gated failure: armed gate released, workers joined, exception propagated");
    return 0;
}

int RunDrain(VulkanDevice& device) {
    {
        std::lock_guard guard(GuestMemory::GpuMutex());
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr, "the drain arm has no active recorder");
        recorder->Sync();
    }
    device.WaitIdle();
    std::puts("drain: the recorder ended with no batch in flight");
    return 0;
}

int RunMode(VulkanDevice& device, const std::string& mode) {
    if (mode == "--locked") return RunLockedControl(device);
    if (mode == "--edge") return RunEdgeOrder(device);
    if (mode == "--unlocked") return RunFreshnessUnlock(device, true);
    if (mode == "--candidate") return RunFreshnessUnlock(device, true);
    if (mode == "--contract") return RunFreshnessUnlock(device, true);
    if (mode == "--hook") return RunFreshnessUnlock(device, false);
    if (mode == "--nested") return RunNestedLock(device);
    if (mode == "--failure") return RunWorkerFailure();
    if (mode == "--gatedfailure") return RunGatedFailure(device);
    if (mode == "--drain") return RunDrain(device);
    std::printf("unknown mode %s\n", mode.c_str());
    return 2;
}

}

int main(int argc, char** argv) {
    try {
        std::vector<std::string> modes;
        for (int index = 1; index < argc; ++index) modes.emplace_back(argv[index]);
        if (modes.empty()) modes = {"--locked", "--edge", "--hook", "--contract", "--nested", "--gatedfailure", "--failure", "--drain"};
#ifdef _WIN32
        _putenv_s("APS5_ASYNC_LANE", "51");
#else
        setenv("APS5_ASYNC_LANE", "51", 1);
#endif
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        for (const auto& mode : modes) {
            const int result = RunMode(*device, mode);
            if (result != 0) return result;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q51 freshness: %s\n", error.what());
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (...) {
        }
        return 1;
    }
}
