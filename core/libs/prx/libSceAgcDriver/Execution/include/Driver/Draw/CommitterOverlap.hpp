#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_COMMITTEROVERLAP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_COMMITTEROVERLAP_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace AgcDriver::DriverDetail {

class CommitterOverlap {
public:
    enum State : std::uint8_t { Between = 0, Idle, GpuWait, DrawResources, DrawOther, DispatchOrOther, DeferredImages, Count };

    struct Snapshot {
        std::array<std::uint64_t, Count> stateNs{};
        std::uint64_t unionNs = 0;
        std::uint64_t waiterNs = 0;
        std::int64_t beginNs = 0;
        std::int64_t endNs = 0;
        std::uint32_t initialWaiters = 0;
        std::uint64_t begins = 0;
        std::uint64_t ends = 0;
        std::uint32_t waiters = 0;
    };

    static bool Enabled() {
        static const bool enabled = std::getenv("APS5_PROFILE_OVERLAP") != nullptr;
        return enabled;
    }

    using Clock = std::int64_t (*)();

    static void InstallTestClock(Clock fn) { clockHook() = fn; }

    static CommitterOverlap& Get() {
        static CommitterOverlap overlap;
        return overlap;
    }

    void EnterCommitterThread() { committerThreadFlag() = true; }
    static bool IsCommitterThread() { return committerThreadFlag(); }

    std::uint32_t OverlapGateValue() const { return gate.load(std::memory_order_acquire); }

    State SetState(State next) {
        if (!Enabled() || !committerThreadFlag()) return Count;
        std::lock_guard lock(mutex);
        account(nowNs());
        const auto previous = state;
        state = next;
        refreshGateLocked();
        return previous;
    }

    void ProducerEnter() {
        if (!Enabled()) return;
        std::lock_guard lock(mutex);
        account(nowNs());
        ++waiters;
        ++begins;
        refreshGateLocked();
    }

    void ProducerExit() {
        if (!Enabled()) return;
        std::lock_guard lock(mutex);
        account(nowNs());
        if (waiters != 0) --waiters;
        ++ends;
        refreshGateLocked();
    }

    Snapshot take() {
        Snapshot snap;
        if (!Enabled()) return snap;
        std::lock_guard lock(mutex);
        account(nowNs());
        snap.stateNs = stateNs;
        snap.unionNs = unionNs;
        snap.waiterNs = waiterNs;
        snap.beginNs = windowBeginNs;
        snap.endNs = lastNs;
        snap.initialWaiters = initialWaiters;
        windowBeginNs = lastNs;
        initialWaiters = waiters;
        snap.begins = begins;
        snap.ends = ends;
        snap.waiters = waiters;
        stateNs = {};
        unionNs = 0;
        waiterNs = 0;
        begins = 0;
        ends = 0;
        return snap;
    }

private:
    static Clock& clockHook() {
        static Clock fn = nullptr;
        return fn;
    }

    static bool& committerThreadFlag() {
        thread_local bool flag = false;
        return flag;
    }

    static std::int64_t nowNs() {
        const Clock fn = clockHook();
        if (fn != nullptr) return fn();
        return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    void account(std::int64_t now) {
        if (!valid) {
            valid = true;
            lastNs = windowBeginNs = now;
            return;
        }
        const std::int64_t delta = now - lastNs;
        if (delta > 0 && waiters != 0) {
            unionNs += static_cast<std::uint64_t>(delta);
            waiterNs += static_cast<std::uint64_t>(delta) * waiters;
            stateNs[state] += static_cast<std::uint64_t>(delta);
        }
        lastNs = now;
    }

    void refreshGateLocked() {
        const bool on = waiters != 0 && state == DeferredImages;
        const std::uint32_t word = gate.load(std::memory_order_relaxed);
        if (static_cast<bool>(word & 1u) == on) return;
        gate.store(((word + 2u) & ~1u) | (on ? 1u : 0u), std::memory_order_release);
    }

    std::mutex mutex;
    std::atomic<std::uint32_t> gate = 0;
    State state = Between;
    std::uint32_t waiters = 0;
    std::int64_t lastNs = 0;
    bool valid = false;
    std::array<std::uint64_t, Count> stateNs{};
    std::uint64_t unionNs = 0;
    std::uint64_t waiterNs = 0;
    std::int64_t windowBeginNs = 0;
    std::uint32_t initialWaiters = 0;
    std::uint64_t begins = 0;
    std::uint64_t ends = 0;
};

class CommitterOverlapScope {
public:
    explicit CommitterOverlapScope(CommitterOverlap::State state)
        : previous(CommitterOverlap::Enabled() ? CommitterOverlap::Get().SetState(state) : CommitterOverlap::Count) {}

    CommitterOverlapScope(const CommitterOverlapScope&) = delete;
    CommitterOverlapScope& operator=(const CommitterOverlapScope&) = delete;

    ~CommitterOverlapScope() {
        if (previous != CommitterOverlap::Count) CommitterOverlap::Get().SetState(previous);
    }

private:
    CommitterOverlap::State previous;
};

}  // namespace AgcDriver::DriverDetail

#endif