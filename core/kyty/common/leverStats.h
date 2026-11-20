#ifndef KYTY_COMMON_LEVER_STATS_H_
#define KYTY_COMMON_LEVER_STATS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#if defined(ANYPS5_KYTY)
namespace Common::LeverStats {

enum class Phase : uint8_t { Elsewhere, Resolve, Materialize, Walk };

enum Line : uint8_t { LineClean, LineGpuBytes, LineImage, LineUnbacked };

constexpr uint32_t Lines            = 4;
constexpr uint32_t Uses             = 4;
constexpr uint32_t Phases           = 4;
constexpr uint32_t RedoKinds        = 2;
constexpr uint32_t RedoClasses      = 4;
constexpr uint32_t RedoFields       = 6;
constexpr uint32_t FaultAccesses    = 2;
constexpr uint32_t FaultPaths       = 10;
constexpr uint32_t TemplateOutcomes = 15;

enum RedoKind : uint32_t { RedoDraw, RedoDispatch };
enum RedoClass : uint32_t { RedoPlain, RedoHeap, RedoNoModule, RedoSlow };
enum RedoField : uint32_t { RedoCount, RedoStages, RedoTotalNs, RedoProgramNs, RedoMaterializeNs, RedoWalkNs };

enum FaultAccess : uint32_t { FaultRead, FaultWrite };
enum FaultPath : uint32_t {
	PathAround,
	PathWriteBack,
	PathFeedback,
	PathQueue,
	PathFlushOpen,
	PathFlushPending,
	PathFlushOther,
	PathTexture,
	PathPlain,
	PathUnhandled
};
enum FaultMark : uint32_t {
	MarkAround       = 1u << 0u,
	MarkWriteBack    = 1u << 1u,
	MarkFeedback     = 1u << 2u,
	MarkQueue        = 1u << 3u,
	MarkFlush        = 1u << 4u,
	MarkOpenWrite    = 1u << 5u,
	MarkPendingWrite = 1u << 6u,
	MarkTexture      = 1u << 7u,
};

enum TemplateOutcome : uint32_t {
	TemplateHit,
	TemplatePartial,
	TemplateMiss,
	TemplateRefusedKey,
	TemplateOff,
	TemplateNoKey,
	TemplateMesh,
	TemplateHeapProgram,
	TemplateHeapImage,
	TemplateTable,
	TemplateWritten,
	TemplateDynamicMip,
	TemplateManyImages,
	TemplateHeapPipeline,
	TemplateStages
};

enum Slot : uint32_t {
	StopFirst,
	StopFirstEnd = StopFirst + Lines * Uses - 1u,
	StopAllClean,
	StopAllCleanEnded,
	StopDirty,
	StopOtherKind,
	StopSingle,
	StopSingleClean,
	StopLoads,
	StopLoadsClean,
	Redo,
	RedoEnd = Redo + RedoKinds * RedoClasses * RedoFields - 1u,
	RedoWalks,
	RedoProbeLoads,
	RedoAroundLoads,
	RedoDirectLoads,
	FaultPhaseCount,
	FaultPhaseCountEnd = FaultPhaseCount + FaultAccesses * Phases - 1u,
	FaultPhaseNs,
	FaultPhaseNsEnd = FaultPhaseNs + FaultAccesses * Phases - 1u,
	FaultPathCount,
	FaultPathCountEnd = FaultPathCount + FaultAccesses * FaultPaths - 1u,
	FaultPathNs,
	FaultPathNsEnd = FaultPathNs + FaultAccesses * FaultPaths - 1u,
	FaultGuardedCopy,
	Template,
	TemplateEnd = Template + TemplateOutcomes - 1u,
	CommitTemplate,
	CommitFull,
	CommitFullCaptured,
	SyncPackets,
	FetchWaits,
	FetchWaitNs,
	FetchReads,
	FetchReadNs,
	CopyDrains,
	CopyDrainNs,
	QueuePops,
	QueuePopNs,
	Slots
};

struct ThreadState {
	bool     redo           = false;
	bool     slow           = false;
	bool     heap           = false;
	bool     no_module      = false;
	Phase    phase          = Phase::Elsewhere;
	uint32_t fault_marks    = 0;
	uint32_t stages         = 0;
	uint32_t walks          = 0;
	uint32_t probe_loads    = 0;
	uint32_t around_loads   = 0;
	uint32_t direct_loads   = 0;
	int64_t  program_ns     = 0;
	int64_t  materialize_ns = 0;
	int64_t  walk_ns        = 0;
	int64_t  inner_walk_ns  = 0;
};

namespace Detail {
inline std::atomic<uint64_t> g_slots[Slots] {};
inline thread_local ThreadState t_state;
}

[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_AHEAD_STOP_STATS");
		return value != nullptr && value[0] == '1' && value[1] == 0;
	}();
	return enabled;
}

[[nodiscard]] inline ThreadState& State() noexcept {
	return Detail::t_state;
}

[[nodiscard]] inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

inline void Add(uint32_t slot, uint64_t value = 1) noexcept {
	Detail::g_slots[slot].fetch_add(value, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Take(uint32_t slot) noexcept {
	return Detail::g_slots[slot].exchange(0, std::memory_order_relaxed);
}

inline void MarkFault(uint32_t marks) noexcept {
	if (Enabled()) {
		State().fault_marks |= marks;
	}
}

class PhaseScope final {
public:
	PhaseScope(Phase phase, bool enabled) noexcept {
		if (!enabled || !State().redo) {
			return;
		}
		auto& state = State();
		m_active    = true;
		m_phase     = phase;
		m_previous  = state.phase;
		state.phase = phase;
		m_start     = NowNs();
	}
	~PhaseScope() {
		if (!m_active) {
			return;
		}
		auto&      state   = State();
		const auto elapsed = NowNs() - m_start;
		state.phase        = m_previous;
		if (m_phase == Phase::Materialize) {
			state.materialize_ns += elapsed;
		} else {
			state.walk_ns += elapsed;
			if (m_previous == Phase::Materialize) {
				state.inner_walk_ns += elapsed;
			}
		}
	}
	PhaseScope(const PhaseScope&)            = delete;
	PhaseScope& operator=(const PhaseScope&) = delete;

private:
	bool    m_active   = false;
	Phase   m_phase    = Phase::Elsewhere;
	Phase   m_previous = Phase::Elsewhere;
	int64_t m_start    = 0;
};

class Timer final {
public:
	Timer(uint32_t count, uint32_t duration, bool enabled) noexcept
	    : m_count(count), m_duration(duration), m_start(enabled ? NowNs() : 0) {}
	~Timer() {
		if (m_start != 0) {
			Add(m_count);
			Add(m_duration, static_cast<uint64_t>(NowNs() - m_start));
		}
	}
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;

private:
	uint32_t m_count;
	uint32_t m_duration;
	int64_t  m_start;
};

}
#endif

#endif
