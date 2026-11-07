#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_COMMITTEROVERLAP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_COMMITTEROVERLAP_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace AgcDriver::DriverDetail {

class CommitterOverlap {
public:
    enum State : std::uint8_t { Between = 0, Idle, GpuWait, DrawResources, DrawRecord, Dispatch, Count };

    struct Snapshot {
        std::array<std::uint64_t, Count> stateNs{};
        std::uint64_t unionNs = 0;
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

    void SetState(State next) {
        if (!Enabled() || !committerThreadFlag()) return;
        std::lock_guard lock(mutex);
        account(nowNs());
        state = next;
    }

    void ProducerEnter() {
        if (!Enabled()) return;
        std::lock_guard lock(mutex);
        account(nowNs());
        ++waiters;
        ++begins;
    }

    void ProducerExit() {
        if (!Enabled()) return;
        std::lock_guard lock(mutex);
        account(nowNs());
        if (waiters != 0) --waiters;
        ++ends;
    }

    Snapshot take() {
        Snapshot snap;
        if (!Enabled()) return snap;
        std::lock_guard lock(mutex);
        account(nowNs());
        snap.stateNs = stateNs;
        snap.unionNs = unionNs;
        snap.begins = begins;
        snap.ends = ends;
        snap.waiters = waiters;
        stateNs = {};
        unionNs = 0;
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
            lastNs = now;
            return;
        }
        const std::int64_t delta = now - lastNs;
        if (delta > 0 && waiters != 0) {
            unionNs += static_cast<std::uint64_t>(delta);
            stateNs[state] += static_cast<std::uint64_t>(delta);
        }
        lastNs = now;
    }

    std::mutex mutex;
    State state = Between;
    std::uint32_t waiters = 0;
    std::int64_t lastNs = 0;
    bool valid = false;
    std::array<std::uint64_t, Count> stateNs{};
    std::uint64_t unionNs = 0;
    std::uint64_t begins = 0;
    std::uint64_t ends = 0;
};

}  // namespace AgcDriver::DriverDetail

#endif