#include "prx/libSceAgcDriver/Graphics/include/PipelineCreationGate.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

using AgcDriver::Graphics::PipelineCreationGate;

namespace {

constexpr std::uint64_t MiB = 1ull << 20u;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

PipelineCreationGate::Settings Settings(std::uint64_t budget, std::uint64_t small = 0) {
    PipelineCreationGate::Settings settings;
    settings.bytesPerWord = 1024;
    settings.budgetBytes = budget;
    settings.smallBytes = small;
    return settings;
}

bool AdmittedWithin(std::atomic<bool>& admitted, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!admitted.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return admitted.load();
}

template <typename TPredicate>
bool Await(TPredicate&& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    return predicate();
}

template <typename TValue>
bool Ready(std::future<TValue>& future) {
    return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

void oversizedCannotBeOvertaken() {
    PipelineCreationGate gate(Settings(8 * MiB, 2 * MiB));
    auto current = gate.Admit(4 * 1024, true);
    std::promise<void> release;
    auto released = release.get_future();
    std::promise<void> admitted;
    auto started = admitted.get_future();
    std::thread oversized([&] {
        const auto ticket = gate.Admit(16 * 1024, true);
        admitted.set_value();
        released.wait();
    });
    const bool queued = Await([&] { return gate.Totals().waited == 1; });
    std::uint64_t overtaken = 0;
    bool ranAlone = false;
    bool completed = queued;
    for (std::uint32_t index = 0; queued && index < 8; ++index) {
        std::promise<PipelineCreationGate::Ticket> next;
        auto nextTicket = next.get_future();
        std::thread newcomer([&] { next.set_value(gate.Admit(4 * 1024, true)); });
        const bool progressed = Await([&] { return Ready(nextTicket) || gate.Totals().waited == 2; });
        if (progressed && Ready(nextTicket)) {
            current = nextTicket.get();
            ++overtaken;
            newcomer.join();
            continue;
        }
        current = {};
        completed = progressed && Await([&] { return Ready(started); });
        const auto stats = gate.Totals();
        ranAlone = stats.inFlight == 1 && stats.inFlightBytes == 16 * MiB;
        release.set_value();
        const bool nextCompleted = Await([&] { return Ready(nextTicket); });
        completed = completed && nextCompleted;
        if (nextCompleted) current = nextTicket.get();
        newcomer.join();
        break;
    }
    current = {};
    if (overtaken != 0 || !queued) {
        completed = Await([&] { return Ready(started); }) && completed;
        release.set_value();
    }
    oversized.join();
    const auto stats = gate.Totals();
    std::printf("oversized FIFO: %llu later budget-limited admissions overtook it, peak %llu jobs / %llu MiB estimated\n", static_cast<unsigned long long>(overtaken), static_cast<unsigned long long>(stats.peakInFlight), static_cast<unsigned long long>(stats.peakBytes / MiB));
    Require(queued && completed, "an oversized waiter or its successor did not complete");
    Require(overtaken == 0, "later budget-limited creations repeatedly overtook the oversized waiter");
    Require(ranAlone && stats.inFlight == 0 && stats.inFlightBytes == 0, "the queued oversized creation did not run alone and release its ticket");
}

void waitersKeepOrder() {
    PipelineCreationGate gate(Settings(8 * MiB, 2 * MiB));
    auto blocker = gate.Admit(8 * 1024, true);
    std::promise<void> firstStarted;
    auto firstReady = firstStarted.get_future();
    std::promise<void> firstRelease;
    auto firstReleased = firstRelease.get_future();
    std::promise<void> secondStarted;
    auto secondReady = secondStarted.get_future();
    std::promise<void> secondRelease;
    auto secondReleased = secondRelease.get_future();
    std::thread first([&] {
        const auto ticket = gate.Admit(6 * 1024, true);
        firstStarted.set_value();
        firstReleased.wait();
    });
    const bool firstQueued = Await([&] { return gate.Totals().waited == 1; });
    std::thread second([&] {
        const auto ticket = gate.Admit(3 * 1024, true);
        secondStarted.set_value();
        secondReleased.wait();
    });
    const bool bothQueued = Await([&] { return gate.Totals().waited == 2; });
    blocker = {};
    const bool eitherStarted = Await([&] { return Ready(firstReady) || Ready(secondReady); });
    const bool firstWasFirst = Ready(firstReady) && !Ready(secondReady);
    firstRelease.set_value();
    secondRelease.set_value();
    first.join();
    second.join();
    const auto stats = gate.Totals();
    Require(firstQueued && bothQueued && eitherStarted && firstWasFirst, "budget-limited waiters were not admitted in arrival order");
    Require(stats.admitted == 3 && stats.waited == 2 && stats.peakBytes == 8 * MiB && stats.inFlight == 0, "FIFO waiters exceeded the budget or leaked tickets");
    std::puts("budget-limited waiters keep their order within the same budget");
}

void exemptionsBypassWaiters() {
    PipelineCreationGate gate(Settings(8 * MiB, 2 * MiB));
    auto blocker = gate.Admit(8 * 1024, true);
    std::promise<void> admitted;
    auto ready = admitted.get_future();
    std::thread waiter([&] {
        const auto ticket = gate.Admit(16 * 1024, true);
        admitted.set_value();
    });
    const bool queued = Await([&] { return gate.Totals().waited == 1; });
    bool bypassed = false;
    {
        const auto small = gate.Admit(2 * 1024, true);
        const auto locked = gate.Admit(6 * 1024, false);
        const auto stats = gate.Totals();
        bypassed = stats.admitted == 3 && stats.unwaited == 2 && stats.waited == 1 && stats.inFlightBytes == 16 * MiB && !Ready(ready);
    }
    blocker = {};
    const bool completed = Await([&] { return Ready(ready); });
    waiter.join();
    Require(queued && bypassed && completed && gate.Totals().inFlight == 0, "small or non-waiting creations were blocked by the waiter queue");
    std::puts("small and non-waiting creations still bypass queued budget-limited waiters");
}

void ticketOwnership() {
    PipelineCreationGate gate(Settings(8 * MiB));
    {
        auto first = gate.Admit(3 * 1024, true);
        auto second = gate.Admit(4 * 1024, true);
        auto moved = std::move(first);
        Require(gate.Totals().inFlight == 2, "moving a ticket released it early");
        second = std::move(moved);
        Require(gate.Totals().inFlight == 1 && gate.Totals().inFlightBytes == 3 * MiB, "move assignment did not release the replaced ticket");
    }
    Require(gate.Totals().inFlight == 0 && gate.Totals().inFlightBytes == 0, "ticket destruction did not release exactly once");
    try {
        const auto ticket = gate.Admit(4 * 1024, true);
        throw std::runtime_error("ticket scope unwind");
    } catch (const std::runtime_error&) {
    }
    Require(gate.Totals().inFlight == 0 && gate.Totals().inFlightBytes == 0, "unwinding did not release the ticket");
    std::puts("moved tickets and exceptional scope exit release their admissions exactly once");
}

void failedWaiterLeavesQueue() {
    auto settings = Settings(8 * MiB);
    std::uint32_t queries = 0;
    settings.availableBytes = [&] {
        if (++queries == 2) throw std::runtime_error("memory query failed");
        return 1024 * MiB;
    };
    PipelineCreationGate gate(settings);
    auto blocker = gate.Admit(8 * 1024, true);
    bool failed = false;
    try {
        const auto ticket = gate.Admit(4 * 1024, true);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    blocker = {};
    {
        const auto recovered = gate.Admit(4 * 1024, true);
        Require(failed && gate.Totals().admitted == 2 && gate.Totals().inFlight == 1, "a failed wait left an admission or waiter behind");
    }
    Require(gate.Totals().inFlight == 0, "the creation after a failed waiter did not release its ticket");
    std::puts("a failed memory query removes its waiter without blocking later admissions");
}

void aloneAndQueued() {
    PipelineCreationGate gate(Settings(8 * MiB));
    auto first = gate.Admit(16 * 1024, true);
    Require(gate.Totals().inFlight == 1 && gate.Totals().inFlightBytes == 16 * MiB, "a creation over the budget did not run alone");
    std::atomic<bool> admitted{false};
    std::thread second([&] {
        const auto ticket = gate.Admit(4 * 1024, true);
        admitted = true;
    });
    const bool early = AdmittedWithin(admitted, std::chrono::milliseconds(200));
    first = {};
    const bool late = AdmittedWithin(admitted, std::chrono::milliseconds(5000));
    second.join();
    const auto stats = gate.Totals();
    Require(!early, "a creation was admitted while the one in flight took the whole budget");
    Require(late && stats.waited == 1 && stats.admitted == 2 && stats.inFlight == 0 && stats.inFlightBytes == 0, "the waiting creation was not admitted once the first finished");
    Require(stats.peakInFlight == 1 && stats.peakBytes == 16 * MiB, "the peak counts more than one creation at once");
    std::puts("a creation over the budget runs alone; the next one waits for it");
}

void fitsTogether() {
    PipelineCreationGate gate(Settings(8 * MiB));
    const auto first = gate.Admit(3 * 1024, true);
    const auto second = gate.Admit(4 * 1024, true);
    const auto stats = gate.Totals();
    Require(stats.waited == 0 && stats.inFlight == 2 && stats.peakBytes == 7 * MiB, "two creations inside the budget did not run together");
    std::puts("creations that fit the budget run together");
}

void smallAndLocked() {
    PipelineCreationGate gate(Settings(8 * MiB, 2 * MiB));
    const auto big = gate.Admit(8 * 1024, true);
    const auto small = gate.Admit(2 * 1024, true);
    const auto locked = gate.Admit(6 * 1024, false);
    const auto stats = gate.Totals();
    Require(stats.waited == 0 && stats.unwaited == 2 && stats.inFlight == 3 && stats.inFlightBytes == 16 * MiB, "a small creation or one under the GPU mutex waited for the budget");
    std::puts("small creations and creations that may not wait run past the budget and are counted");
}

void availableMemory() {
    auto settings = Settings(1024 * MiB);
    std::atomic<std::uint64_t> available{2 * MiB};
    settings.availableBytes = [&] { return available.load(); };
    PipelineCreationGate gate(settings);
    const auto first = gate.Admit(1024, true);
    std::atomic<bool> admitted{false};
    std::thread second([&] {
        const auto ticket = gate.Admit(2 * 1024, true);
        admitted = true;
    });
    const bool early = AdmittedWithin(admitted, std::chrono::milliseconds(400));
    available = 512 * MiB;
    const bool late = AdmittedWithin(admitted, std::chrono::milliseconds(5000));
    second.join();
    Require(!early, "a creation was admitted past half the available memory");
    Require(late && gate.Totals().waited == 1, "the creation was not admitted once memory became available");
    std::puts("half the available memory bounds the creations in flight");
}

void disabled() {
    PipelineCreationGate gate(Settings(0));
    const auto first = gate.Admit(1u << 20u, true);
    const auto second = gate.Admit(1u << 20u, true);
    const auto stats = gate.Totals();
    Require(stats.admitted == 0 && stats.inFlight == 0, "a zero budget did not turn the gate off");
    std::puts("a zero budget turns the gate off");
}

void defaults() {
    const auto settings = PipelineCreationGate::SettingsFromEnvironment();
    const auto physical = PipelineCreationGate::PhysicalMemoryBytes();
    Require(physical != 0 && PipelineCreationGate::AvailableMemoryBytes() != 0, "physical or available memory reads zero");
    if (std::getenv("APS5_PIPELINE_COMPILE_BUDGET_MIB") == nullptr) Require(settings.budgetBytes == (physical / 8u >> 20u << 20u) && settings.availableBytes != nullptr, "the default budget is not an eighth of physical memory bounded by the available memory");
    std::printf("default budget %llu MiB of %llu MiB physical, %llu bytes per word, small below %llu MiB\n", static_cast<unsigned long long>(settings.budgetBytes >> 20u), static_cast<unsigned long long>(physical >> 20u), static_cast<unsigned long long>(settings.bytesPerWord), static_cast<unsigned long long>(settings.smallBytes >> 20u));
}

}

int main() {
    try {
        aloneAndQueued();
        fitsTogether();
        smallAndLocked();
        availableMemory();
        disabled();
        defaults();
        oversizedCannotBeOvertaken();
        waitersKeepOrder();
        exemptionsBypassWaiters();
        ticketOwnership();
        failedWaiterLeavesQueue();
        std::puts("pipeline creation gate tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pipeline creation gate test failed: %s\n", error.what());
        return 1;
    }
}
