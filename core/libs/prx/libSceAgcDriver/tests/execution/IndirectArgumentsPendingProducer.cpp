#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <SDL_loadso.h>
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <initializer_list>
#include <iostream>
#include <new>
#include <memory>
#include <atomic>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

alignas(256) std::array<std::uint32_t, 7> AddressWriterCode{0x7e000202u, 0x7e020203u, 0x7e040204u, 0x7e060280u, 0xdc7c8000u, 0x00000003u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 7> CounterCode{0xf4080200u, 0xfa000000u, 0xbf8cc07fu, 0x7e000281u, 0xe0c80000u, 0x80020000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 15> SpinCode{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};

constexpr std::uint32_t Groups = 5;
constexpr std::array<std::uint32_t, 3> Stale{3u, 288u, 0xffffffffu};
constexpr std::uint32_t SpinIterations = 1u << 20u;
constexpr std::uint32_t SpinDispatches = 16;
constexpr std::uint32_t ThreadDimensions = 0x8061u;
constexpr std::uint32_t GroupCounts = 0x8041u;
constexpr std::uint32_t Compute = 0x20u;
constexpr std::uint32_t OtherCompute = 0x28u;
constexpr std::uint32_t Graphics = 0u;
constexpr std::size_t GuestBytes = 1u << 20u;

struct Slot {
    alignas(256) std::array<std::uint32_t, 4> arguments{};
    alignas(256) std::array<std::uint32_t, 4> counter{};
    alignas(256) std::array<std::uint32_t, 4> counterDescriptor{};
    alignas(256) std::array<std::uint32_t, 4> spin{};
    alignas(256) std::array<std::uint32_t, 4> spinDescriptor{};
    alignas(256) std::uint32_t label = 0;
    alignas(256) std::uint32_t done = 0;
};

enum class Order { SeparateSubmissions, OneSubmission, LabelWaitInOneSubmission, LabelWaitBetweenSubmissions, LabelWaitAcrossQueues, ArgumentLabelInEarlierBatch };

struct Variant {
    const char* name;
    std::uint32_t initiator;
    Order order;
};

constexpr std::array<Variant, 6> Variants{{
    {"thread dimensions, separate submissions", ThreadDimensions, Order::SeparateSubmissions},
    {"thread dimensions, one submission", ThreadDimensions, Order::OneSubmission},
    {"thread dimensions, label wait in one submission", ThreadDimensions, Order::LabelWaitInOneSubmission},
    {"thread dimensions, label wait between submissions", ThreadDimensions, Order::LabelWaitBetweenSubmissions},
    {"thread dimensions, label wait across compute queues", ThreadDimensions, Order::LabelWaitAcrossQueues},
    {"group counts read on the CPU behind an argument label of an earlier graphics batch", GroupCounts, Order::ArgumentLabelInEarlierBatch},
}};

std::uint32_t Low(const volatile void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const volatile void* pointer) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer)) >> 32u);
}

std::array<std::uint32_t, 4> Descriptor(const void* pointer, std::uint32_t bytes) {
    return {Low(pointer), High(pointer) & 0xffffu, bytes, 0x31016facu};
}

std::uint32_t Load(const std::uint32_t& value) {
    return reinterpret_cast<const volatile std::uint32_t&>(value);
}

void SetShaderRegister(std::vector<std::uint32_t>& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Program(std::vector<std::uint32_t>& words, const void* code, std::initializer_list<std::uint32_t> userData) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code));
    SetShaderRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetShaderRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    for (const std::uint32_t offset : {0x207u, 0x208u, 0x209u}) SetShaderRegister(words, offset, 1);
    SetShaderRegister(words, 0x212, 0);
    SetShaderRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetShaderRegister(words, offset++, value);
}

void WriteData(std::vector<std::uint32_t>& words, std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00100200u, Low(target), High(target), value});
}

void WaitEqual(std::vector<std::uint32_t>& words, std::uint32_t* target, std::uint32_t value) {
    words.insert(words.end(), {0xc0053c00u, 0x13u, Low(target), High(target), value, 0xffffffffu, 0x19u});
}

void Spin(std::vector<std::uint32_t>& words, Slot& slot, std::uint32_t dispatches) {
    for (std::uint32_t index = 0; index < dispatches; ++index) {
        Program(words, SpinCode.data(), {Low(slot.spinDescriptor.data()), High(slot.spinDescriptor.data()), SpinIterations});
        words.insert(words.end(), {0xc0031500u, 1, 1, 1, GroupCounts});
    }
}

void Produce(std::vector<std::uint32_t>& words, Slot& slot) {
    Program(words, AddressWriterCode.data(), {Low(slot.arguments.data()), High(slot.arguments.data()), Groups, 1u, 1u});
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, GroupCounts});
}

void Consume(std::vector<std::uint32_t>& words, Slot& slot, std::uint32_t initiator) {
    Program(words, CounterCode.data(), {Low(slot.counterDescriptor.data()), High(slot.counterDescriptor.data())});
    words.insert(words.end(), {0xc0021102u, 1u, Low(slot.arguments.data()), High(slot.arguments.data())});
    words.insert(words.end(), {0xc0011600u, 0u, initiator});
}

void Submit(std::uint32_t queue, std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    if (queue == Graphics) Require(sceAgcDriverSubmitDcb(&packet) == 0, "graphics submit failed");
    else Require(sceAgcDriverSubmitAcb(queue, &packet) == 0, "compute submit failed");
}

ShaderRecompiler::RecompileResult Recompile(AgcDriver::VulkanDevice& device, Slot& slot, std::span<const std::uint32_t> code, std::initializer_list<std::uint32_t> userData) {
    const std::array<ShaderRecompiler::MemoryRegion, 2> memory{{
        {reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)},
        {reinterpret_cast<std::uintptr_t>(slot.counterDescriptor.data()), std::as_bytes(std::span(slot.counterDescriptor))},
    }};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{1, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, std::span(userData.begin(), userData.size()), compute, std::nullopt, std::nullopt, memory},
        device.ComputeTarget(32),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

class ProducerSubmissionGate {
public:
    explicit ProducerSubmissionGate(AgcDriver::VulkanDevice& device) : device(device) {
#ifdef _WIN32
        constexpr const char* libraryName = "vulkan-1.dll";
#else
        constexpr const char* libraryName = "libvulkan.so.1";
#endif
        library.reset(SDL_LoadObject(libraryName));
        Require(library != nullptr, "cannot load Vulkan for the producer submission gate");
        resolve = reinterpret_cast<PFN_vkGetDeviceProcAddr>(SDL_LoadFunction(library.get(), "vkGetDeviceProcAddr"));
        Require(resolve != nullptr, "cannot resolve vkGetDeviceProcAddr for the producer submission gate");
        const auto createSemaphore = function<PFN_vkCreateSemaphore>("vkCreateSemaphore");
        destroySemaphore = function<PFN_vkDestroySemaphore>("vkDestroySemaphore");
        signalSemaphore = reinterpret_cast<PFN_vkSignalSemaphore>(resolve(device.Device(), "vkSignalSemaphoreKHR"));
        if (signalSemaphore == nullptr) signalSemaphore = function<PFN_vkSignalSemaphore>("vkSignalSemaphore");
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        Require(createSemaphore(device.Device(), &info, nullptr, &semaphore) == VK_SUCCESS, "cannot create the producer submission gate");
    }

    ~ProducerSubmissionGate() {
        if (worker.joinable()) {
            worker.request_stop();
            worker.join();
        }
        Release();
        try {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            device.WaitIdle();
        } catch (...) {
        }
        destroySemaphore(device.Device(), semaphore, nullptr);
    }

    void Arm(AgcDriver::Graphics::Recorder& recorder) {
        recorder.Commands();
        recorder.HoldNextSubmit(semaphore, 1);
    }

    void ReleaseAfterSubmission(AgcDriver::Graphics::Recorder& recorder, std::uint64_t serial) {
        worker = std::jthread([this, &recorder, serial](std::stop_token stop) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!stop.stop_requested() && recorder.NewestSubmitted() < serial && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::microseconds(100));
            producerSubmitted = recorder.NewestSubmitted() >= serial;
            Release();
            if (!producerSubmitted && !stop.stop_requested()) std::fprintf(stderr, "pending producer gate: watchdog expired with producer batch %llu unsubmitted (vkSignalSemaphore %d)\n", static_cast<unsigned long long>(serial), static_cast<int>(releaseResult));
        });
    }

    void ReleaseAfterGpuMutex(const std::atomic<bool>* waitEntered = nullptr) {
        worker = std::jthread([this, waitEntered](std::stop_token stop) {
            std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex(), std::defer_lock);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline &&
                   ((waitEntered != nullptr && !waitEntered->load()) || !lock.try_lock())) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            acquiredGpuMutex = lock.owns_lock();
            if (acquiredGpuMutex && waitEntered == nullptr) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            Release();
        });
    }

    void FinishUnlockedWait() {
        const bool releasedBeforeReturn = signaled.load();
        worker.join();
        Require(releaseResult == VK_SUCCESS, "cannot release the submitted writer gate");
        Require(acquiredGpuMutex, "the submitted writer wait held the GPU mutex");
        Require(releasedBeforeReturn, "the submitted writer wait returned before the gate was released");
    }

    void Finish() {
        worker.join();
        Require(releaseResult == VK_SUCCESS, "cannot release the producer submission gate");
        Require(producerSubmitted, "the CPU argument read did not submit the later leased producer before waiting for its earlier argument label");
        device.WaitIdle();
    }

    VkResult Release() {
        if (signaled.exchange(true)) return releaseResult;
        VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        info.semaphore = semaphore;
        info.value = 1;
        releaseResult = signalSemaphore(device.Device(), &info);
        return releaseResult;
    }

private:
    template <typename TFunction>
    TFunction function(const char* name) const {
        const auto function = reinterpret_cast<TFunction>(resolve(device.Device(), name));
        Require(function != nullptr, std::string("cannot resolve ") + name + " for the producer submission gate");
        return function;
    }

    AgcDriver::VulkanDevice& device;
    std::unique_ptr<void, decltype(&SDL_UnloadObject)> library{nullptr, &SDL_UnloadObject};
    PFN_vkGetDeviceProcAddr resolve = nullptr;
    PFN_vkDestroySemaphore destroySemaphore = nullptr;
    PFN_vkSignalSemaphore signalSemaphore = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    std::jthread worker;
    std::atomic<bool> signaled{false};
    bool producerSubmitted = false;
    bool acquiredGpuMutex = false;
    VkResult releaseResult = VK_NOT_READY;
};

int RunSubmittedWritePresync(AgcDriver::VulkanDevice& device) {
    using AgcDriver::Graphics::Recorder;
    auto& gpu = AgcDriver::GuestMemory::GpuMutex();
    std::unique_lock lock(gpu);
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the submitted writer test has no active recorder");
    if (!recorder->AsyncLaneActive() || !recorder->HasTimeline()) {
        std::puts("skipped: the submitted writer test needs both timeline lanes");
        return VulkanTestSkipped;
    }
    device.WaitIdle();

    constexpr std::uint32_t written = 42;
    std::uint32_t value = 0;
    std::uint32_t valueAtLabel = 0;
    std::uint32_t completions = 0;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&value));
    const std::array ranges{std::pair{address, address + sizeof(value)}};
    ProducerSubmissionGate gate(device);
    {
        const Recorder::LaneScope async(1);
        gate.Arm(*recorder);
        recorder->NotePendingWrite(address, sizeof(value));
        recorder->OnComplete([&] {
            value = written;
            ++completions;
        });
        recorder->OnComplete([&] {
            valueAtLabel = value;
            ++completions;
        });
        recorder->Submit();
    }

    const auto pending = recorder->DescribePendingWrite(address, sizeof(value));
    Require(pending.has_value() && !pending->open && !pending->signaled, "the gated writer was not submitted and unsignaled");
    gate.ReleaseAfterGpuMutex();
    recorder->WaitSubmittedWrites(std::move(lock), ranges);
    gate.FinishUnlockedWait();

    std::lock_guard reapLock(gpu);
    const auto afterWait = recorder->DescribePendingWrite(address, sizeof(value));
    Require(afterWait.has_value() && afterWait->serial == pending->serial, "the wait retired the writer batch");
    Require(completions == 0 && value == 0 && valueAtLabel == 0, "the wait ran a write-back completion");
    recorder->SyncThrough(address, sizeof(value));
    Require(completions == 2 && value == written && valueAtLabel == written, "the locked sync did not run the write-back completions in order");
    Require(!recorder->DescribePendingWrite(address, sizeof(value)).has_value(), "the locked sync left the writer pending");
    std::puts("submitted writer presync released the GPU mutex and preserved completion order");
    return 0;
}

PFN_vkGetDeviceProcAddr actualDeviceProc = nullptr;
std::atomic<bool> bdaWaitEntered{false};

VKAPI_ATTR VkResult VKAPI_CALL ObserveBdaWait(VkDevice device, const VkSemaphoreWaitInfo* wait, std::uint64_t timeout) {
    bdaWaitEntered.store(true);
    const auto real = reinterpret_cast<PFN_vkWaitSemaphoresKHR>(actualDeviceProc(device, "vkWaitSemaphoresKHR"));
    return real(device, wait, timeout);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ObserveBdaDeviceProc(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkWaitSemaphoresKHR") return reinterpret_cast<PFN_vkVoidFunction>(&ObserveBdaWait);
    return actualDeviceProc(device, name);
}

int RunCachedBdaPresync(AgcDriver::VulkanDevice& device, Slot& slot) {
    using AgcDriver::Graphics::AddressSpaceCounters;
    using AgcDriver::Graphics::GuestBufferMemory;
    using AgcDriver::Graphics::Recorder;
    auto context = device.graphicsContext();
    if (context.hostImportAlignment == 0 || context.asyncQueue == VK_NULL_HANDLE || !device.CanWaitUnlocked()) {
        std::puts("skipped: cached BDA presync requires host imports and both timeline lanes");
        return VulkanTestSkipped;
    }
    auto& gpu = AgcDriver::GuestMemory::GpuMutex();
    std::unique_lock lock(gpu);
    device.WaitIdle();

    std::uint32_t valueBetweenLabels = 0;
    actualDeviceProc = context.deviceProc;
    context.deviceProc = &ObserveBdaDeviceProc;
    Recorder recorder(context, true);
    Require(recorder.AsyncLaneActive(), "the cached BDA test could not create its async timeline");
    recorder.Activate();
    {
        GuestBufferMemory seed(context);
        seed.AcquireRegistered();
        seed.Upload(true);
        seed.WriteBack();
    }
    recorder.Sync();

    constexpr std::uint32_t first = 42;
    constexpr std::uint32_t second = 43;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&slot.label));
    ProducerSubmissionGate gate(device);
    {
        const Recorder::LaneScope async(1);
        gate.Arm(recorder);
        recorder.AfterCompletions(address, std::as_bytes(std::span(&first, 1)), 1, Compute, false);
        recorder.OnComplete([&] { valueBetweenLabels = slot.label; });
        recorder.AfterCompletions(address, std::as_bytes(std::span(&second, 1)), 2, Compute, false);
        recorder.Submit();
    }
    const auto pending = recorder.DescribePendingWrite(address, sizeof(slot.label));
    Require(pending.has_value() && !pending->open && !pending->signaled, "the cached BDA producer was not submitted and unsignaled");
    bdaWaitEntered.store(false);
    gate.ReleaseAfterGpuMutex(&bdaWaitEntered);
    lock.unlock();
    try {
        GuestBufferMemory probe(context);
        probe.PresyncSubmittedWrites(recorder, true);
        gate.FinishUnlockedWait();
        Require(bdaWaitEntered.load(), "the cached BDA preflight did not enter the Vulkan timeline wait");
    } catch (...) {
        lock.lock();
        throw;
    }
    lock.lock();
    Require(slot.label == 0 && valueBetweenLabels == 0 && Recorder::PendingCompletionLabels() == 2, "the cached BDA preflight completed the pending labels");
    const auto before = AddressSpaceCounters().hits;
    {
        GuestBufferMemory consumer(context);
        consumer.AcquireRegistered();
        Require(AddressSpaceCounters().hits == before + 1, "the BDA consumer did not use the cached address space");
        consumer.Upload(true);
        consumer.WriteBack();
    }
    Require(valueBetweenLabels == first && slot.label == second && Recorder::PendingCompletionLabels() == 0, "locked BDA upload did not complete the labels in order");
    recorder.Sync();
    std::puts("cached BDA presync released the GPU mutex and preserved completion order");
    return 0;
}

int RunBdaUnlockedOrder(AgcDriver::VulkanDevice& device) {
    using namespace AgcDriver::Graphics;
    auto& gpu = AgcDriver::GuestMemory::GpuMutex();
    std::unique_lock lock(gpu);
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "the BDA ordering test has no active recorder");
    const auto context = device.graphicsContext();
    if (!recorder->AsyncLaneActive() || context.queue == context.asyncQueue || RuntimeSwitches::Set(RuntimeSwitch::LaneSerialize)) {
        std::puts("skipped: BDA ordering needs two independent Vulkan lanes");
        return VulkanTestSkipped;
    }
    recorder->Sync();
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    Require(fill != nullptr, "cannot resolve vkCmdFillBuffer for the BDA ordering test");

    constexpr std::uint32_t written = 42;
    constexpr std::uint32_t signaled = 43;
    constexpr std::uint32_t peerWritten = 44;
    enum class Order { Locked, Unlocked, ExplicitSync, ExplicitWriters, OrderedModel };
    struct Values {
        std::uint32_t value = 0;
        std::uint32_t label = 0;
        std::uint32_t valueAtLabel = 0;
        bool callbackRan = false;
    };
    for (const auto order : {Order::Locked, Order::Unlocked, Order::ExplicitSync, Order::ExplicitWriters, Order::OrderedModel}) {
        const bool mayPublishEarly = order != Order::Locked && order != Order::OrderedModel;
        const auto values = std::make_shared<Values>();
        const auto address = reinterpret_cast<std::uintptr_t>(&values->value);
        const auto labelAddress = reinterpret_cast<std::uintptr_t>(&values->label);
        const auto bytes = sizeof(values->value);
        const auto staging = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::memset(staging->Bytes().data(), 0, bytes);
        const auto peerStaging = order == Order::OrderedModel ? std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT) : nullptr;
        ProducerSubmissionGate gate(device);
        {
            const Recorder::LaneScope writerLane(1);
            gate.Arm(*recorder);
            fill(recorder->Commands(), staging->Handle(), 0, bytes, written);
            recorder->OweHostRead();
            recorder->NotePendingWrite(address, bytes);
            recorder->OnComplete([values, staging, bytes] { std::memcpy(&values->value, staging->Bytes().data(), bytes); });
        }
        {
            const Recorder::LaneScope labelLane(0);
            static_cast<void>(recorder->Commands());
            recorder->OnComplete([values] {
                values->valueAtLabel = values->value;
                values->callbackRan = true;
            });
            recorder->AfterCompletions(labelAddress, std::as_bytes(std::span(&signaled, 1)), 1, Compute, false);
            recorder->SubmitAllLanes();
        }
        const auto first = recorder->DescribePendingWrite(address, bytes);
        auto last = recorder->DescribePendingWrite(labelAddress, bytes);
        Require(first && last && !first->open && !last->open && first->serial < last->serial && !first->signaled, "the first writer was not submitted behind the gate before the later label");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!last->signaled && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
            last = recorder->DescribePendingWrite(labelAddress, bytes);
        }
        if (!last->signaled) {
            Require(gate.Release() == VK_SUCCESS, "cannot release the BDA ordering gate");
            recorder->Sync();
            Require(values->value == written && values->label == signaled, "the independent-progress probe did not finish after releasing the gate");
            std::puts("skipped: the later Vulkan lane did not signal while the earlier lane was gated");
            return VulkanTestSkipped;
        }
        Require(values->value == 0 && values->label == 0 && !values->callbackRan, "a completion landed before reaping either fence");
        bool peerFenceSignaled = false;
        if (order != Order::Locked) {
            std::exception_ptr failure;
            std::promise<void> peerFinished;
            const auto peerDone = peerFinished.get_future();
            bool watchdogExpired = false;
            VkResult releaseResult = VK_NOT_READY;
            std::jthread release([&] {
                watchdogExpired = peerDone.wait_for(std::chrono::seconds(10)) != std::future_status::ready;
                releaseResult = gate.Release();
            });
            std::uint64_t peerSerial = 0;
            std::jthread peer([&] {
                try {
                    std::lock_guard peerLock(gpu);
                    const auto blocked = recorder->DescribePendingWrite(address, bytes);
                    Require(blocked && !blocked->signaled, "the first writer escaped the gate before the peer reaped");
                    if (mayPublishEarly) {
                        if (order == Order::ExplicitSync) {
                            recorder->SyncThrough(labelAddress, bytes, false);
                        } else if (order == Order::ExplicitWriters) {
                            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> ranges{{{labelAddress, labelAddress + bytes}}};
                            const auto wait = recorder->FinishWritersByLane(ranges, 0, 0);
                            Require(wait.targets[0] == last->serial && wait.targets[1] == 0, "explicit writer wait selected the wrong batches");
                        } else {
                            recorder->Reap();
                        }
                        Require(values->value == 0 && values->label == signaled && values->callbackRan && values->valueAtLabel == 0, "per-lane retirement did not expose the later label before the gated write-back");
                    } else {
                        const auto orderedReap = [&] {
                            const auto earlier = recorder->DescribePendingWrite(address, bytes);
                            if (earlier && !earlier->signaled) return;
                            if (earlier) recorder->FinishUpTo(earlier->serial);
                            recorder->Reap();
                        };
                        orderedReap();
                        const auto deferred = recorder->DescribePendingWrite(labelAddress, bytes);
                        Require(deferred && deferred->signaled && Recorder::PendingCompletionLabels() != 0 && !values->callbackRan && values->label == 0, "the ordered model did not leave the signaled label pending");

                        const Recorder::LaneScope peerLane(0);
                        const auto peerAddress = reinterpret_cast<std::uintptr_t>(peerStaging->Bytes().data());
                        fill(recorder->Commands(), peerStaging->Handle(), 0, bytes, peerWritten);
                        recorder->OweHostRead();
                        recorder->NotePendingWrite(peerAddress, bytes);
                        recorder->NotePendingWrite(address, bytes);
                        recorder->OnComplete([values, peerStaging, bytes] { std::memcpy(&values->value, peerStaging->Bytes().data(), bytes); });
                        recorder->Submit();
                        auto pendingPeer = recorder->DescribePendingWrite(peerAddress, bytes);
                        Require(pendingPeer && !pendingPeer->open && pendingPeer->serial > last->serial, "the peer did not submit independent GPU work while the prefix was held");
                        peerSerial = pendingPeer->serial;
                        const auto peerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                        while (!pendingPeer->signaled && std::chrono::steady_clock::now() < peerDeadline) {
                            std::this_thread::yield();
                            pendingPeer = recorder->DescribePendingWrite(peerAddress, bytes);
                        }
                        peerFenceSignaled = pendingPeer->signaled;
                        Require(values->value == 0 && !values->callbackRan && values->label == 0, "the ordered model released CPU publication before the earlier writer");
                        if (peerFenceSignaled) {
                            std::uint32_t peerValue = 0;
                            std::memcpy(&peerValue, peerStaging->Bytes().data(), bytes);
                            Require(peerValue == peerWritten, "the peer fence signaled before the independent GPU store was visible");
                        }
                    }
                } catch (...) {
                    failure = std::current_exception();
                }
                peerFinished.set_value();
            });
            lock.unlock();
            std::exception_ptr waitFailure;
            try {
                recorder->WaitSerial(last->serial);
            } catch (...) {
                waitFailure = std::current_exception();
            }
            peer.join();
            release.join();
            lock.lock();
            if (waitFailure) std::rethrow_exception(waitFailure);
            if (failure) std::rethrow_exception(failure);
            Require(!watchdogExpired && releaseResult == VK_SUCCESS, "the peer did not finish before the watchdog released the earlier writer");
            recorder->FinishUpTo(last->serial);
            if (peerSerial != 0) {
                Require(values->value == written && values->valueAtLabel == written, "the reserved prefix did not publish the earlier writer before the label");
                const auto newerWriter = recorder->DescribePendingWrite(address, bytes);
                Require(newerWriter && newerWriter->serial == peerSerial, "the resource recheck missed the writer recorded while the mutex was unlocked");
                recorder->SyncThrough(address, bytes, false);
                Require(values->value == peerWritten && !recorder->DescribePendingWrite(address, bytes), "the recheck did not make the newer write visible");
                std::printf("BDA recheck after relock: original prefix %u, current resource %u\n", written, values->value);
            }
        } else {
            Require(gate.Release() == VK_SUCCESS, "cannot release the BDA ordering gate");
            recorder->Sync();
        }
        Require(values->value == (order == Order::OrderedModel ? peerWritten : written) && values->label == signaled && values->callbackRan && values->valueAtLabel == (mayPublishEarly ? 0u : written), "retirement did not produce the expected label visibility order");
        const char* name = "ordered retirement model";
        if (order == Order::Locked) name = "locked Sync";
        if (order == Order::Unlocked) name = "unlocked interleaving";
        if (order == Order::ExplicitSync) name = "explicit SyncThrough";
        if (order == Order::ExplicitWriters) name = "explicit FinishWritersByLane";
        std::printf("BDA %s, serials %llu/%llu: write-back %u, label %u, write-back seen by label %u\n", name, static_cast<unsigned long long>(first->serial), static_cast<unsigned long long>(last->serial), values->value, values->label, values->valueAtLabel);
        if (order == Order::OrderedModel) std::printf("BDA ordered retirement model: peer acquired the GPU mutex and %s independent GPU work before releasing the earlier writer\n", peerFenceSignaled ? "completed" : "submitted");
    }
    return 0;
}

void RunArgumentLabel(Slot& slot) {
    AgcDriverShutdown_nid_postfix();
    slot.arguments = {3, 1, 1, 0};
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "cannot open the argument-label test device");
    const auto producer = Recompile(*device, slot, AddressWriterCode, {Low(slot.arguments.data()), High(slot.arguments.data()), Groups, 1, 1});
    const auto consumer = Recompile(*device, slot, CounterCode, {Low(slot.counterDescriptor.data()), High(slot.counterDescriptor.data())});
    const auto preparedProducer = device->PrepareDispatch(producer, {}, 1, reinterpret_cast<std::uintptr_t>(AddressWriterCode.data()));
    const auto preparedConsumer = device->PrepareDispatch(consumer, {}, 0, reinterpret_cast<std::uintptr_t>(CounterCode.data()));
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    Require(recorder != nullptr, "the argument-label test has no active recorder");
    ProducerSubmissionGate gate(*device);
    gate.Arm(*recorder);
    const std::uint32_t labelValue = 2;
    const auto labelAddress = reinterpret_cast<std::uintptr_t>(&slot.arguments[1]);
    Require(device->WriteLabelOnGpu(labelAddress, std::as_bytes(std::span(&labelValue, 1)), 1, Graphics, false) == 0, "the argument label was not recorded on the GPU");
    device->SubmitRecorded(false);
    const auto producerSerial = recorder->OpenSerial();
    gate.ReleaseAfterSubmission(*recorder, producerSerial);
    const auto deferredBefore = AgcDriver::Graphics::LeaseCounters().deferred;
    device->Dispatch(producer, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(AddressWriterCode.data()), preparedProducer);
    Require(AgcDriver::Graphics::LeaseCounters().deferred > deferredBefore, "the address-based producer did not defer its registry lease");
    Require(recorder->Submissions() < producerSerial && recorder->Recording(), "the producer was submitted before the CPU argument decision");
    Require(recorder->PendingLabel(labelAddress, 4, 0).has_value(), "the earlier argument label retired before the CPU argument decision");
    const auto outcome = device->DispatchIndirect(consumer, reinterpret_cast<std::uintptr_t>(slot.arguments.data()), {}, reinterpret_cast<std::uintptr_t>(CounterCode.data()), preparedConsumer);
    Require(outcome.cpuReason == 2, "the indirect dispatch did not take the pending-label CPU fallback (reason " + std::to_string(outcome.cpuReason) + ")");
    std::fprintf(stderr, "pending producer CPU fallback: cpuReason %d\n", outcome.cpuReason);
    gate.Finish();
    slot.done = 1;
}

void Run(Slot& slot, const Variant& variant) {
    slot.arguments = {Stale[0], Stale[1], Stale[2], 0};
    slot.counter = {};
    slot.counterDescriptor = Descriptor(slot.counter.data(), sizeof(slot.counter));
    slot.spinDescriptor = Descriptor(slot.spin.data(), sizeof(slot.spin));
    slot.label = 0;
    slot.done = 0;
    std::vector<std::uint32_t> first;
    std::vector<std::uint32_t> second;
    switch (variant.order) {
    case Order::SeparateSubmissions:
        Spin(first, slot, SpinDispatches);
        Produce(first, slot);
        Consume(second, slot, variant.initiator);
        WriteData(second, &slot.done, 1);
        Submit(Compute, first);
        Submit(Compute, second);
        break;
    case Order::OneSubmission:
        Spin(first, slot, SpinDispatches);
        Produce(first, slot);
        Consume(first, slot, variant.initiator);
        WriteData(first, &slot.done, 1);
        Submit(Compute, first);
        break;
    case Order::LabelWaitInOneSubmission:
        Spin(first, slot, SpinDispatches);
        Produce(first, slot);
        WriteData(first, &slot.label, 1);
        WaitEqual(first, &slot.label, 1);
        Consume(first, slot, variant.initiator);
        WriteData(first, &slot.done, 1);
        Submit(Compute, first);
        break;
    case Order::LabelWaitBetweenSubmissions:
        Spin(first, slot, SpinDispatches);
        Produce(first, slot);
        WriteData(first, &slot.label, 1);
        WaitEqual(second, &slot.label, 1);
        Consume(second, slot, variant.initiator);
        WriteData(second, &slot.done, 1);
        Submit(Compute, first);
        Submit(Compute, second);
        break;
    case Order::LabelWaitAcrossQueues:
        Spin(first, slot, SpinDispatches);
        Produce(first, slot);
        WriteData(first, &slot.label, 1);
        WaitEqual(second, &slot.label, 1);
        Consume(second, slot, variant.initiator);
        WriteData(second, &slot.done, 1);
        Submit(OtherCompute, first);
        Submit(Compute, second);
        break;
    case Order::ArgumentLabelInEarlierBatch:
        RunArgumentLabel(slot);
        break;
    }
    if (variant.order != Order::ArgumentLabelInEarlierBatch) AgcDriverWaitIdle_nid_postfix();
    const auto start = std::chrono::steady_clock::now();
    while (Load(slot.done) == 0) {
        Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), std::string(variant.name) + ": the consumer's completion label did not land");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    Require(Load(slot.arguments[0]) == Groups && Load(slot.arguments[1]) == 1 && Load(slot.arguments[2]) == 1, std::string(variant.name) + ": the producer's address-based store did not write the indirect arguments");
    Require(Load(slot.counter[0]) == Groups, std::string(variant.name) + ": the indirect dispatch ran " + std::to_string(Load(slot.counter[0])) + " threads, expected the " + std::to_string(Groups) + " its producer's address-based store wrote");
}

}

int main(int argc, char** argv) {
    try {
        const bool submittedWritePresync = argc == 2 && std::string_view(argv[1]) == "--submitted-write-presync";
        const bool cachedBdaPresync = argc == 2 && std::string_view(argv[1]) == "--cached-bda-presync";
        const bool bdaUnlockedOrder = argc == 2 && std::string_view(argv[1]) == "--bda-unlocked-order";
        if (submittedWritePresync || cachedBdaPresync || bdaUnlockedOrder) {
#ifdef _WIN32
            _putenv_s("APS5_ASYNC_LANE", "51");
#else
            setenv("APS5_ASYNC_LANE", "51", 1);
#endif
        }
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (submittedWritePresync) return RunSubmittedWritePresync(*device);
        if (bdaUnlockedOrder) return RunBdaUnlockedOrder(*device);
        if (!cachedBdaPresync) device.reset();
#ifdef _WIN32
        auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, GuestBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        auto* block = static_cast<std::uint8_t*>(std::aligned_alloc(GuestBytes, GuestBytes));
#endif
        Require(block != nullptr, "cannot allocate guest memory");
        GuestAllocations::Mutation().Add(block, GuestBytes, true, true, true);
        auto* slots = new (block) Slot[Variants.size()];
        if (cachedBdaPresync) return RunCachedBdaPresync(*device, slots[0]);
        for (std::size_t index = 0; index < Variants.size(); ++index) {
            if (argc > 1 && std::to_string(index) != argv[1]) continue;
            Run(slots[index], Variants[index]);
            std::cout << Variants[index].name << ": passed\n";
        }
        std::cout << "indirect arguments pending producer tests passed\n";
        AgcDriverShutdown_nid_postfix();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try {
            AgcDriverShutdown_nid_postfix();
        } catch (const std::exception& shutdownError) {
            std::cerr << shutdownError.what() << '\n';
        }
        return 1;
    }
}
