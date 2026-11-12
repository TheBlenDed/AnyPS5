#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/SubmissionRing.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#ifdef _WIN32
#include <windows.h>
#endif
#include <immintrin.h>
#include <algorithm>
#include <tuple>
#include <vector>

namespace AgcDriver::DriverDetail {

void PollSleep() {
#ifdef _WIN32
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (timer != nullptr) {
        LARGE_INTEGER due{};
        due.QuadPart = -2000;
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(200));
}

int Driver::waitTimeoutMs() {
    static const int value = [] { const char* text = std::getenv("APS5_GPU_WAIT_TIMEOUT_MS"); return text ? std::atoi(text) : 1000; }();
    return value;
}

bool Driver::pollReapAll() {
    static const bool all = std::getenv("APS5_POLL_REAP") != nullptr;
    return all;
}

bool Driver::pollTryEach() {
    static const bool each = std::getenv("APS5_POLL_TRY_EACH") != nullptr;
    return each;
}

bool Driver::traceLateLabels() {
    static const bool trace = std::getenv("APS5_TRACE_LATE_LABELS") != nullptr;
    return trace;
}

std::vector<Driver::PendingWait>& Driver::pendingWaits() {
    struct WaitsTag {};
    return HostThreadLocal<std::vector<PendingWait>, WaitsTag>();
}

void Driver::notePendingWait(std::span<const std::uint32_t> packet, std::uint32_t queue, std::uint64_t received) {
    if (packet.size() > std::tuple_size_v<decltype(PendingWait::words)>) return;
    static const bool allQueues = std::getenv("APS5_LAND_WAITS_ALL_QUEUES") != nullptr;
    if (!allQueues && queue == 0) return;
    if (Pm4::WaitSatisfiedUnchecked(packet)) return;
    auto& waits = pendingWaits();
    for (const auto& wait : waits) {
        if (wait.count == packet.size() && std::equal(packet.begin(), packet.end(), wait.words.begin())) return;
    }
    PendingWait wait{};
    std::copy(packet.begin(), packet.end(), wait.words.begin());
    wait.count = packet.size();
    wait.received = received;
    waits.push_back(wait);
}

void Driver::landPendingWaits(std::uint32_t queue) {
    auto& waits = pendingWaits();
    if (waits.empty()) return;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    struct LandStats {
        std::uint64_t lands = 0, waited = 0;
        double waitedMs = 0, maxMs = 0;
        std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
    };
    static thread_local LandStats stats;
    const PacketHistory none{};
    for (const auto& wait : waits) {
        const auto packet = std::span<const std::uint32_t>(wait.words).first(wait.count);
        ++stats.lands;
        if (Pm4::WaitSatisfiedUnchecked(packet)) continue;
        const auto started = std::chrono::steady_clock::now();
        waitMemory(packet, queue, none, wait.received, false, true);
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        ++stats.waited;
        stats.waitedMs += ms;
        stats.maxMs = std::max(stats.maxMs, ms);
    }
    waits.clear();
    if (profile && std::chrono::steady_clock::now() - stats.reported > std::chrono::seconds(10)) {
        AgcDriver::ProfilePrint_nid_no_patch("[landwait] queue 0x%x (10 s): %llu waits landed before a guest read, %llu had to wait for the GPU: %.1f ms in total, max %.2f ms\n", queue, static_cast<unsigned long long>(stats.lands), static_cast<unsigned long long>(stats.waited), stats.waitedMs, stats.maxMs);
        stats = LandStats{};
    }
}

namespace {

enum class WaitDiagReason : std::uint32_t {
    ValueSatisfied, CpuStore, LabelTakenTable, LabelTakenPending, WritesSettled
};

enum class WaitDiagAction : std::uint32_t {
    None, OpenWriteSubmitted, LabelDueSubmitted, Reaped, WritesFlushed
};

const char* WaitDiagReasonName(WaitDiagReason reason) {
    switch (reason) {
        case WaitDiagReason::ValueSatisfied: return "value_satisfied";
        case WaitDiagReason::CpuStore: return "cpu_store";
        case WaitDiagReason::LabelTakenTable: return "label_taken_table";
        case WaitDiagReason::LabelTakenPending: return "label_taken_pending";
        case WaitDiagReason::WritesSettled: return "writes_settled";
    }
    return "unknown";
}

const char* WaitDiagActionName(WaitDiagAction action) {
    switch (action) {
        case WaitDiagAction::None: return "none";
        case WaitDiagAction::OpenWriteSubmitted: return "open_write_submitted";
        case WaitDiagAction::LabelDueSubmitted: return "label_due_submitted";
        case WaitDiagAction::Reaped: return "reaped";
        case WaitDiagAction::WritesFlushed: return "gpu_writes_flushed";
    }
    return "unknown";
}

bool waitDiagEnabled() {
    static const bool enabled = std::getenv("APS5_WAIT_TRACE") != nullptr;
    return enabled;
}

struct WaitDiagLane {
    std::atomic<std::uint64_t> waits{0}, over5ms{0}, blockedUs{0}, maxUs{0};
};

std::array<WaitDiagLane, 65>& waitDiagLanes() {
    static std::array<WaitDiagLane, 65> lanes{};
    return lanes;
}

std::atomic<std::int64_t>& waitDiagNextReportNs() {
    static std::atomic<std::int64_t> next{0};
    return next;
}

void reportWaitDiagLanes() {
    const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto next = waitDiagNextReportNs().load(std::memory_order_relaxed);
    if (next != 0 && nowNs < next) return;
    if (!waitDiagNextReportNs().compare_exchange_strong(next, nowNs + std::int64_t{10} * 1000 * 1000 * 1000)) return;
    if (next == 0) return;
    auto& lanes = waitDiagLanes();
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const auto waits = lanes[i].waits.exchange(0, std::memory_order_relaxed);
        const auto over = lanes[i].over5ms.exchange(0, std::memory_order_relaxed);
        const auto blockedUs = lanes[i].blockedUs.exchange(0, std::memory_order_relaxed);
        const auto maxUs = lanes[i].maxUs.exchange(0, std::memory_order_relaxed);
        if (waits == 0) continue;
        char name[8];
        if (i + 1 == lanes.size()) std::snprintf(name, sizeof(name), "other");
        else std::snprintf(name, sizeof(name), "0x%x", static_cast<unsigned>(i));
        AgcDriver::ProfilePrint_nid_no_patch("[waitdiag] queue %s (10 s): waitdiag_waits %llu, waitdiag_over5ms %llu, waitdiag_blocked_ms %.1f, waitdiag_max_ms %.1f\n", name,
                                             static_cast<unsigned long long>(waits), static_cast<unsigned long long>(over), static_cast<double>(blockedUs) / 1000.0, static_cast<double>(maxUs) / 1000.0);
    }
}

}

void Driver::waitMemory(std::span<const std::uint32_t> packet, std::uint32_t queue, const PacketHistory& context, std::uint64_t received, bool heldAtSubmit, bool requireMemory) {

    auto start = std::chrono::steady_clock::now();
    auto lastDone = packetsDone.load();

    GuestMemory::TagGpuLockThread(queue);
    auto& outcomes = waitOutcomes();
    const bool wide = ((packet[0] >> 8u) & 0xffu) == 0x93u;
    const std::size_t awaitedBytes = Pm4::WaitAwaitedBytes(packet);
    const auto alignMask = awaitedBytes == 8 ? ~7ull : ~3ull;
    const std::uint64_t awaited = (packet[2] & static_cast<std::uint32_t>(alignMask)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f queue 0x%x waits 0x%llx == 0x%x (now 0x%x)\n", TraceMs(), queue, static_cast<unsigned long long>(awaited), packet[4],
                               *reinterpret_cast<const volatile std::uint32_t*>(awaited));
    struct WaitTrace {
        bool enabled; std::uint32_t queue; std::uint64_t address; std::chrono::steady_clock::time_point begin;
        ~WaitTrace() {
            if (!enabled) return;
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            if (ms >= 0.5) std::fprintf(stderr, "[gpu] %.1f queue 0x%x wait on 0x%llx done after %.1f ms\n", TraceMs(), queue, static_cast<unsigned long long>(address), ms);
        }
    } waitTrace{traceGpu, queue, awaited, std::chrono::steady_clock::now()};
    bool warned = false;

    struct EpochPoint {
        bool bump = true;
        ~EpochPoint() {
            if (bump) bumpEpoch(&EpochBumps::waits);
        }
    } epochPoint;
    GuestMemory::CheckRange(reinterpret_cast<const void*>(awaited), awaitedBytes, awaitedBytes);
    const bool waitDiag = waitDiagEnabled();
    std::uint64_t diagEntryValue = 0;
    if (waitDiag) diagEntryValue = awaitedBytes == 8 ? *reinterpret_cast<const volatile std::uint64_t*>(awaited) : *reinterpret_cast<const volatile std::uint32_t*>(awaited);
    WaitDiagReason diagReason = WaitDiagReason::ValueSatisfied;
    WaitDiagAction diagLast = WaitDiagAction::None;
    std::uint64_t diagSubmits = 0, diagReaps = 0, diagFlushes = 0;
    if (Pm4::WaitSatisfiedUnchecked(packet)) {
        ++outcomes.atEntry;
        return;
    }
    if (heldAtSubmit) return;
    if (std::uint32_t writer = queue; !requireMemory && storedSince(packet, awaited, awaitedBytes, received, &writer)) {
        if (writer != queue) notePendingWait(packet, queue, received);
        else epochPoint.bump = false;
        return;
    }

    static const bool labelShortcut = std::getenv("APS5_NO_LABEL_SHORTCUT") == nullptr;
    const bool labelTable = labelShortcut && !requireMemory;
    static const bool overlapSubmit = std::getenv("APS5_NO_WAIT_OVERLAP_SUBMIT") == nullptr;

    static const bool waitLock = std::getenv("APS5_WAIT_LOCK") != nullptr;

    std::uint64_t refusedGeneration = 0;

    bool behindCompletion = false;
    const auto traceLate = [&](const char* outcome, const std::optional<Graphics::Recorder::LabelHit>& hit, Graphics::Recorder::LabelRefusal refusal) {
        static std::atomic<int> shown{0};
        if (shown.fetch_add(1) >= 200) return;
        std::fprintf(stderr, "[late] queue 0x%x wait on 0x%llx (function %u ref 0x%x mask 0x%x) received %llu: %s; entry value 0x%llx stamp %llu generation %llu, refusal %d\n", queue, static_cast<unsigned long long>(awaited), packet[1] & 7u, packet[4], packet[5], static_cast<unsigned long long>(received), outcome,
                     static_cast<unsigned long long>(hit ? hit->value : 0), static_cast<unsigned long long>(hit ? hit->stamp : 0), static_cast<unsigned long long>(hit ? hit->generation : 0), static_cast<int>(refusal));
    };
    const auto takeLabel = [&](const std::optional<Graphics::Recorder::LabelHit>& hit, Graphics::Recorder::LabelRefusal refusal, bool polling, bool unlockedHit) {
        behindCompletion = refusal == Graphics::Recorder::LabelRefusal::BehindCompletion;
        const bool trace = traceLateLabels() && (refusal != Graphics::Recorder::LabelRefusal::None || (hit.has_value() && hit->late));
        if (!hit.has_value() || !Pm4::WaitComparesValue(packet, hit->value)) {
            if (trace) traceLate(hit.has_value() ? "value mismatch" : "refused by the table", hit, refusal);
            return false;
        }
        if (hit->late) {
            if (hit->generation == refusedGeneration) return false;
            if (!GuestMemory::UnchangedSinceCollected(awaited, awaitedBytes, hit->generation)) {
                ++outcomes.lateRefusedCpuStore;
                refusedGeneration = hit->generation;
                if (trace) traceLate("refused, CPU store since the group closed", hit, refusal);
                return false;
            }
            ++outcomes.fromRecorderLate;
            if (trace) traceLate("trusted", hit, refusal);
        }
        ++(polling ? outcomes.fromRecorderPolling : outcomes.fromRecorder);
        if (unlockedHit) ++outcomes.fromRecorderUnlocked;
        // Answered from the label table before the store ran on the GPU: with the async compute lane,
        // a store recorded in the other lane must also precede this queue's later work there.
        if (Graphics::Recorder::LaneOfQueue(hit->queue) != Graphics::Recorder::LaneOfQueue(queue)) Graphics::Recorder::NoteConsumedLabel(awaited, awaitedBytes);
        if (hit->queue == queue) {
            ++outcomes.fromRecorderSameQueue;
            if (!hit->late) epochPoint.bump = false;
        } else {
            notePendingWait(packet, queue, received);
        }
        return true;
    };

    std::uint64_t seenGeneration = Graphics::Recorder::WriteGeneration();

    const bool unlocked = !waitLock && overlapSubmit;
    if (unlocked) {
        if (labelTable) {
            Graphics::Recorder::LabelRefusal refusal{};
            if (takeLabel(Graphics::Recorder::LookupLabel(awaited, awaitedBytes, received, &refusal), refusal, false, true)) return;
        }
    }
    const bool lockedEntry = !unlocked || Graphics::Recorder::SnapshotWriteOverlaps(awaited, awaitedBytes);
    if (!lockedEntry) ++outcomes.entriesUnlocked;

    static const bool blockingEntry = std::getenv("APS5_WAIT_LOCK_ENTRY") != nullptr;
    bool recheck = !lockedEntry;
    bool entryTryFailed = false;
    if (lockedEntry) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Wait);
        std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
        if (blockingEntry || !unlocked) {
            gpuLock.lock();
        } else if (!gpuLock.try_lock()) {
            ++outcomes.entryTriesFailed;
            recheck = true;
            entryTryFailed = true;
        }
        if (const auto localDevice = gpuLock.owns_lock() ? device.Load() : std::shared_ptr<VulkanDevice>{}) {

            if (labelTable) {
                Graphics::Recorder::LabelRefusal refusal{};
                if (takeLabel(localDevice->PendingLabel(awaited, awaitedBytes, received, &refusal), refusal, false, false)) return;
            }

            if (!overlapSubmit || localDevice->OpenWriteOverlaps(awaited, awaitedBytes)) localDevice->SubmitRecorded(false);
        }
    }

    static const bool pollProfile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& poll = pollStats();
    const auto pollRefused = [&](Graphics::Recorder::LabelRefusal refusal) {
        if (pollProfile && refusal != Graphics::Recorder::LabelRefusal::None && static_cast<std::size_t>(refusal) < poll.refusals.size()) ++poll.refusals[static_cast<std::size_t>(refusal)];
    };

    const bool reapAll = pollReapAll() || !labelShortcut || queue == 0;
    std::chrono::steady_clock::time_point lastTry{};
    const auto pollService = [&](bool sleeping) {
        if (pollProfile) ++poll.ticks;
        const auto generation = Graphics::Recorder::WriteGeneration();
        const bool changed = generation != seenGeneration || recheck;
        const bool labelDue = labelFlushDue();
        const bool reap = sleeping && (behindCompletion || ((reapAll || queue0Dormant.load(std::memory_order_relaxed)) && completionsPending()));
        if (!changed && !labelDue && !reap) return false;

        if (!changed && !pollTryEach()) {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastTry < PollTryInterval) return false;
            lastTry = now;
        }
        if (changed && unlocked && labelTable) {

            Graphics::Recorder::LabelRefusal refusal{};
            if (takeLabel(Graphics::Recorder::LookupLabel(awaited, awaitedBytes, received, &refusal), refusal, true, true)) {
                if (pollProfile) ++poll.tableHits;
                diagReason = WaitDiagReason::LabelTakenTable;
                return true;
            }
            pollRefused(refusal);
        }
        if (pollProfile) ++poll.tries;
        std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::try_to_lock);
        if (!gpuLock.owns_lock()) {
            ++triesFailed[TryPoll];
            if (pollProfile) ++poll.triesFailed;
            return false;
        }

        struct TryHold {
            bool enabled;
            PollStats& poll;
            std::chrono::steady_clock::time_point start;
            ~TryHold() {
                if (!enabled) return;
                const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                poll.tryHoldUs += us;
                poll.longestTryHoldUs = std::max(poll.longestTryHoldUs, us);
            }
        } tryHold{pollProfile, poll, pollProfile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
        const auto localDevice = device.Load();
        if (localDevice == nullptr) {
            seenGeneration = generation;
            recheck = false;
            return false;
        }
        bool submitted = false;
        if (changed) {
            seenGeneration = generation;
            recheck = false;
            if (labelTable) {
                Graphics::Recorder::LabelRefusal refusal{};
                if (takeLabel(localDevice->PendingLabel(awaited, awaitedBytes, received, &refusal), refusal, true, false)) {
                    if (pollProfile) ++poll.tableHits;
                    diagReason = WaitDiagReason::LabelTakenPending;
                    return true;
                }
                pollRefused(refusal);
            }
            if (localDevice->OpenWriteOverlaps(awaited, awaitedBytes)) {
                localDevice->SubmitRecorded(false);
                submitted = true;
                diagLast = WaitDiagAction::OpenWriteSubmitted;
            }
        }
        if (labelDue && !submitted && Graphics::Recorder::PendingLabelSince().has_value()) {
            localDevice->SubmitRecorded(false);
            submitted = true;
            diagLast = WaitDiagAction::LabelDueSubmitted;
        }
        if (submitted) {
            ++outcomes.pollSubmits;
            ++diagSubmits;
            if (pollProfile) ++poll.submits;
        }
        if (reap) {
            const auto withWorkBefore = pollProfile ? Graphics::Recorder::ReapsWithWork() : 0;
            localDevice->ReapRecorded();
            ++outcomes.pollReaps;
            ++diagReaps;
            diagLast = WaitDiagAction::Reaped;
            if (pollProfile) {
                ++poll.reaps;
                ++(behindCompletion ? poll.reapsBehindCompletion : poll.reapsOther);
                poll.reapsWithWork += Graphics::Recorder::ReapsWithWork() - withWorkBefore;
            }
        }
        return false;
    };

    if (entryTryFailed && pollService(false)) return;

    noteWaitFields(queue, packet);
    noteWaitBlocked(queue, awaited, true);
    struct BlockedWait {
        Driver& driver;
        std::uint32_t queue;
        ~BlockedWait() { driver.noteWaitBlocked(queue, 0, false); }
    } blocked{*this, queue};
    static const bool pauseSpin = std::getenv("APS5_NO_PAUSE_SPIN") == nullptr;
    static const bool storeHeldResults = std::getenv("APS5_NO_QUEUED_DRAW_READ_ORDER") == nullptr;
    const auto spinLimit = queue == 0 ? std::chrono::microseconds(1500) : std::chrono::microseconds(100);
    const auto spinStart = std::chrono::steady_clock::now();
    bool spinning = pauseSpin;
    std::uint32_t polls = 0;
    diagReason = WaitDiagReason::ValueSatisfied;
    diagLast = WaitDiagAction::None;
    diagSubmits = 0;
    diagReaps = 0;
    diagFlushes = 0;
    struct WaitDiagPrint {
        bool enabled;
        std::uint32_t queue;
        std::span<const std::uint32_t> packet;
        std::uint64_t awaited;
        std::size_t bytes;
        std::uint64_t entryValue;
        const std::uint32_t* polls;
        WaitDiagReason* reason;
        WaitDiagAction* last;
        std::uint64_t* submits;
        std::uint64_t* reaps;
        std::uint64_t* flushes;
        std::chrono::steady_clock::time_point begin;
        ~WaitDiagPrint() {
            if (!enabled) return;
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            auto& lanes = waitDiagLanes();
            auto& lane = lanes[std::min<std::size_t>(queue, lanes.size() - 1)];
            lane.waits.fetch_add(1, std::memory_order_relaxed);
            const auto us = static_cast<std::uint64_t>(ms * 1000.0);
            lane.blockedUs.fetch_add(us, std::memory_order_relaxed);
            auto maxUs = lane.maxUs.load(std::memory_order_relaxed);
            while (us > maxUs && !lane.maxUs.compare_exchange_weak(maxUs, us, std::memory_order_relaxed)) {
            }
            if (ms >= 5.0) lane.over5ms.fetch_add(1, std::memory_order_relaxed);
            reportWaitDiagLanes();
            if (ms < 5.0) return;
            if (packet.size() < 6) return;
            const bool wideWait = ((packet[0] >> 8u) & 0xffu) == 0x93u;
            const std::uint64_t reference = wideWait ? (packet.size() > 5 ? packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u) : 0u) : packet[4];
            const std::uint64_t mask = wideWait ? (packet.size() > 7 ? packet[6] | (static_cast<std::uint64_t>(packet[7]) << 32u) : 0u) : packet[5];
            const std::uint64_t exitValue = bytes == 8 ? *reinterpret_cast<const volatile std::uint64_t*>(awaited) : *reinterpret_cast<const volatile std::uint32_t*>(awaited);
            char text[1024];
            const auto used = std::snprintf(text, sizeof(text), "[waitdiag] queue 0x%x opcode 0x%x addr 0x%llx bytes %zu func %u ref 0x%llx mask 0x%llx blocked_ms %.1f polls %llu reason %s last %s submits %llu reaps %llu flushes %llu entry 0x%llx exit 0x%llx", queue,
                                             static_cast<unsigned>((packet[0] >> 8u) & 0xffu), static_cast<unsigned long long>(awaited), bytes, packet[1] & 7u, static_cast<unsigned long long>(reference), static_cast<unsigned long long>(mask), ms,
                                             static_cast<unsigned long long>(*polls), WaitDiagReasonName(*reason), WaitDiagActionName(*last), static_cast<unsigned long long>(*submits), static_cast<unsigned long long>(*reaps),
                                             static_cast<unsigned long long>(*flushes), static_cast<unsigned long long>(entryValue), static_cast<unsigned long long>(exitValue));
            if (used < 0) return;
            std::size_t size = std::min(static_cast<std::size_t>(used), sizeof(text) - 4);
            std::size_t shown = 0;
            for (const auto& record : Driver::writeHistory()) {
                if (shown == 3) break;
                if (record.length == 0 || record.target > awaited || awaited >= record.target + std::max<std::uint64_t>(record.length, 4)) continue;
                const auto more = std::snprintf(text + size, sizeof(text) - size, " writer q0x%x:0x%x@0x%llx+0x%llx", record.queue, record.opcode, static_cast<unsigned long long>(record.target),
                                                static_cast<unsigned long long>(record.length));
                if (more <= 0) break;
                size = std::min(size + static_cast<std::size_t>(more), sizeof(text) - 4);
                ++shown;
            }
            text[size++] = '\n';
            text[size] = '\0';
            AgcDriver::ProfilePrint_nid_no_patch("%s", text);
        }
    } waitDiagPrint{waitDiag, queue, packet, awaited, awaitedBytes, diagEntryValue, &polls, &diagReason, &diagLast, &diagSubmits, &diagReaps, &diagFlushes, std::chrono::steady_clock::now()};
    StallWatch stall(queue);
    while (!Pm4::WaitSatisfiedUnchecked(packet)) {
        if (std::uint32_t writer = queue; !requireMemory && storedSince(packet, awaited, awaitedBytes, received, &writer)) {
            if (writer != queue) notePendingWait(packet, queue, received);
            else epochPoint.bump = false;
            diagReason = WaitDiagReason::CpuStore;
            return;
        }
        ++polls;
        if (spinning) {
            _mm_pause();
            if ((polls & 63u) != 0) continue;
            if (std::chrono::steady_clock::now() - spinStart > spinLimit) spinning = false;
        } else if (pauseSpin || polls >= 4000) {
            PollSleep();
        } else {
            std::this_thread::yield();
        }
        CheckFailure();
        if (pollService(!spinning)) return;
        if (storeHeldResults && !spinning && (polls & 31u) == 0) {
            GuestMemory::FlushGpuWrites(awaited, awaitedBytes);
            diagLast = WaitDiagAction::WritesFlushed;
            ++diagFlushes;
        }
        noteAwaitingTitle(queue, awaited);
        if (requireMemory && (polls & 15u) == 0) {
            std::unique_lock settleLock(GuestMemory::GpuMutex(), std::try_to_lock);
            if (settleLock.owns_lock()) {
                if (const auto localDevice = device.Load(); localDevice != nullptr && localDevice->RecordedWritesSettled(awaited, awaitedBytes)) {
                    diagReason = WaitDiagReason::WritesSettled;
                    return;
                }
            }
        }
        if (const auto done = packetsDone.load(); done != lastDone || (!requireMemory && packetsInFlight.load() != 0)) {
            lastDone = done;
            start = std::chrono::steady_clock::now();
            stall.Progressed();
        }
        stall.Check(((packet[0] >> 8u) & 0xffu) == 0x93u ? "WAIT_REG_MEM_64 at" : "WAIT_REG_MEM at", awaited);

        if (!warned && std::chrono::steady_clock::now() - start > std::chrono::milliseconds(waitTimeoutMs())) {
            warned = true;
            ++outcomes.timedOut;
            {
                static std::mutex reportedMutex;
                static std::set<std::uint64_t> reported;
                static std::uint64_t timeouts = 0;
                const std::lock_guard lock(reportedMutex);
                if (++timeouts % 20 == 0) std::fprintf(stderr, "[gpu] %llu GPU waits have timed out\n", static_cast<unsigned long long>(timeouts));
                if (!reported.insert(awaited).second) continue;
            }
            const bool wide = ((packet[0] >> 8u) & 0xffu) == 0x93u;
            const std::uint64_t reference = wide ? packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u) : packet[4];
            const std::uint64_t mask = wide ? packet[6] | (static_cast<std::uint64_t>(packet[7]) << 32u) : packet[5];
            const std::uint64_t current = awaitedBytes == 8 ? *reinterpret_cast<const volatile std::uint64_t*>(awaited) : *reinterpret_cast<const volatile std::uint32_t*>(awaited);
            std::fprintf(stderr, "[gpu] queue 0x%x WAIT_REG_MEM%s%s at 0x%llx timed out after %dms (function %u ref 0x%llx mask 0x%llx value 0x%llx)\n", queue, wide ? "_64" : "", requireMemory ? " (landing)" : "",
                         static_cast<unsigned long long>(awaited), waitTimeoutMs(), packet[1] & 7u, static_cast<unsigned long long>(reference), static_cast<unsigned long long>(mask), static_cast<unsigned long long>(current));
            const auto address = awaited;
            for (const auto& record : writeHistory()) {
                if (record.length != 0 && record.target <= address && address < record.target + std::max<std::uint64_t>(record.length, 4))
                    std::fprintf(stderr, "[gpu]   earlier write by queue 0x%x opcode 0x%x at 0x%llx+0x%llx\n", record.queue, record.opcode, static_cast<unsigned long long>(record.target), static_cast<unsigned long long>(record.length));
            }

            if (const auto reportDevice = device.Load(); reportDevice != nullptr) {
                Graphics::Recorder::LabelRefusal refusal{};
                const auto hit = reportDevice->PendingLabel(awaited, awaitedBytes, received, &refusal);
                const auto since = Graphics::Recorder::PendingLabelSince();
                const auto age = since.has_value() ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *since).count() : -1.0;
                std::fprintf(stderr, "[gpu]   recorder: table %s (refusal %u), open batch writes it %d, snapshot write overlaps %d, pending completion labels %llu, write-back completions %llu, pending label age %.1f ms, write generation %llu seen %llu, queue 0 dormant %d, completions pending %d, label due %d\n",
                             hit.has_value() ? "hit" : "miss", static_cast<unsigned>(refusal), reportDevice->OpenWriteOverlaps(awaited, awaitedBytes) ? 1 : 0, Graphics::Recorder::SnapshotWriteOverlaps(awaited, awaitedBytes) ? 1 : 0,
                             static_cast<unsigned long long>(Graphics::Recorder::PendingCompletionLabels()), static_cast<unsigned long long>(Graphics::Recorder::PendingWriteBackCompletions()), age,
                             static_cast<unsigned long long>(Graphics::Recorder::WriteGeneration()), static_cast<unsigned long long>(seenGeneration), queue0Dormant.load(std::memory_order_relaxed) ? 1 : 0, completionsPending() ? 1 : 0, labelFlushDue() ? 1 : 0);
            }
            context.Print(stderr, "[gpu]   preceding packet %s\n");
        }
    }
    ++outcomes.polled;
}

}
