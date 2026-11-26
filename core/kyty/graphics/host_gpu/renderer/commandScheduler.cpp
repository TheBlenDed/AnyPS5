#include "graphics/host_gpu/renderer/commandScheduler.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/queueCheckpoints.h"
#endif

#include "graphics/host_gpu/renderer/frameCapture.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/focusedGraphicsTiming.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/inPlaceMemory.h"
#if defined(ANYPS5_KYTY)
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#endif

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <tuple>

#if defined(ANYPS5_KYTY)
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>
#endif

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

#if defined(ANYPS5_KYTY)
class CommandScheduler::PassTiming {
public:
	PassTiming(GraphicContext& graphics, const char* trigger, bool trace_submits = false,
	           bool trace_packet_bounds = false, bool capture_combat = false)
	    : m_graphics(graphics), m_trigger(trigger), m_trace_submits(trace_submits),
	      m_passes_per_frame(trace_packet_bounds ? 12288 : 8192), m_trace_packet_bounds(trace_packet_bounds),
	      m_combat_pending(capture_combat) {}

	static int64_t HostTime() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(
		    std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	void BeginHostSubmit(uint64_t frame, uint64_t tick) {
		if (!m_trace_submits || m_pool == nullptr || frame < m_first_frame ||
		    frame - m_first_frame >= FrameCount || m_saved) {
			return;
		}
		const auto begin = HostTime();
		std::scoped_lock lock(m_submit_mutex);
		if (m_submits.empty()) {
			m_submits.reserve(HostSubmitCapacity);
			m_first_submit.store(tick, std::memory_order_relaxed);
		}
		const auto index = tick - m_first_submit.load(std::memory_order_relaxed);
		if (index < HostSubmitCapacity) {
			if (index >= m_submits.size()) {
				m_submits.resize(static_cast<size_t>(index) + 1);
			}
			m_submits[index].frame = frame;
			m_submits[index].begin = begin;
		}
	}

	bool MaybeTraceHostSubmit(uint64_t tick) const {
		const auto first = m_first_submit.load(std::memory_order_relaxed);
		return first != 0 && tick - first < HostSubmitCapacity;
	}

	void PublishHostSubmit(uint64_t tick, int64_t now) {
		std::scoped_lock lock(m_submit_mutex);
		if (auto* sample = FindHostSubmit(tick)) {
			sample->publish = now;
		}
	}

	void ReadyHostSubmit(uint64_t tick, int64_t record_end, int64_t copies_ready) {
		std::scoped_lock lock(m_submit_mutex);
		if (auto* sample = FindHostSubmit(tick)) {
			sample->record_end = record_end;
			sample->copies_ready = copies_ready;
		}
	}

	void QueueHostSubmit(uint64_t tick, int64_t request, int64_t acquired, int64_t returned,
	                     uint32_t waits, bool has_buffer) {
		std::scoped_lock lock(m_submit_mutex);
		if (auto* sample = FindHostSubmit(tick)) {
			sample->queue_request = request;
			sample->queue_lock = acquired;
			sample->queue_return = returned;
			sample->waits = waits;
			sample->has_buffer = has_buffer;
		}
	}

	~PassTiming() {
		m_focus.Close(m_graphics.device);
		if (m_pool != nullptr) {
			m_graphics.device.destroyQueryPool(m_pool, nullptr);
		}
	}

	void Poll(uint64_t frame, auto completed_tick) {
		if (m_saved || (m_armed && (m_pool == nullptr || m_save_attempts >= 3))) {
			if (!m_combat_pending) {
				return;
			}
			AdvanceToCombat();
		}
		if (m_armed && frame < m_first_frame + FrameCount) {
			return;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now < m_next_poll) {
			return;
		}
		m_next_poll = now + std::chrono::milliseconds(100);
		if (m_armed) {
			auto last = m_samples.empty() ? uint64_t {0} : m_samples.back().tick;
			if (!m_submits.empty()) {
				last = std::max(last, m_first_submit.load(std::memory_order_relaxed) + m_submits.size() - 1);
			}
			if (last == 0 || completed_tick() >= last) {
				++m_save_attempts;
				Save();
			}
			return;
		}
		std::error_code error;
		if (!std::filesystem::exists(m_trigger, error)) {
			return;
		}
		m_armed = true;
		const auto families = m_graphics.physical_device.getQueueFamilyProperties();
		const auto family = m_graphics.queue_family;
		if (family >= families.size() || families[family].timestampValidBits == 0) {
			Log::WriteToConsoleAndLog("[kyty-pass-timing] queue has no Vulkan timestamp support\n");
			return;
		}
		const auto bits = families[family].timestampValidBits;
		m_mask = bits >= 64 ? UINT64_MAX : (uint64_t {1} << bits) - 1;
		vk::QueryPoolCreateInfo create {};
		create.queryType = vk::QueryType::eTimestamp;
		create.queryCount = QueryCount();
		if (m_graphics.device.createQueryPool(&create, nullptr, &m_pool) != vk::Result::eSuccess) {
			m_pool = nullptr;
			Log::WriteToConsoleAndLog("[kyty-pass-timing] cannot create Vulkan timestamp pool\n");
			return;
		}
		if (FocusPixelShader()) m_focus.Open(m_graphics.device, m_graphics.physical_device.getFeatures().pipelineStatisticsQuery);
		m_first_frame = frame + 1;
		m_samples.reserve(QueryCount() / 2);
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-pass-timing] armed for renderer flips {} through {}\n",
		    m_first_frame, m_first_frame + FrameCount - 1));
	}

	uint32_t Begin(const CommandRecorder& recorder, uint64_t frame, uint64_t tick,
	               uint64_t shader, const RenderState* render, const char* kind = "compute",
	               uint64_t pixel_shader = 0) {
		if (m_pool == nullptr || frame < m_first_frame || frame - m_first_frame >= FrameCount) {
			return 0;
		}
		const auto frame_index = static_cast<size_t>(frame - m_first_frame);
		if (m_frame_samples[frame_index] >= m_passes_per_frame) {
			++m_dropped[frame_index];
			return 0;
		}
		if (m_samples.empty()) {
			recorder.resetQueryPool(m_pool, QueryCount());
			m_focus.Reset(recorder);
		}
		++m_frame_samples[frame_index];
		const auto query = static_cast<uint32_t>(m_samples.size()) * 2;
		m_samples.push_back({frame, tick, shader, render != nullptr ? "graphics" : kind,
		                     render ? render->width : 0, render ? render->height : 0,
		                     render ? render->num_color_attachments : 0,
		                     render && render->depth_stencil_attachment.has_depth, pixel_shader, false});
		recorder.writePassTimestamp(m_pool, query);
		if (render && pixel_shader && pixel_shader == FocusPixelShader()) m_focus.Begin(recorder, frame, tick, query + 1);
		return query + 1;
	}

	void NoteComputeDispatch(uint32_t token, uint64_t module, const char* dispatch,
	                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
		auto& sample = m_samples[token / 2];
		sample.module = module;
		sample.dispatch = dispatch;
		sample.groups_x = groups_x;
		sample.groups_y = groups_y;
		sample.groups_z = groups_z;
	}

	void NoteGraphicsShader(uint32_t token, uint64_t pixel_shader) {
		auto& sample = m_samples[token / 2];
		if (sample.pixel_shader != pixel_shader) {
			sample.pixel_shader = 0;
			sample.mixed_pixel_shaders = true;
		}
	}

	void End(const CommandRecorder& recorder, uint32_t token) const {
		m_focus.End(recorder, token);
		recorder.writePassTimestamp(m_pool, token);
	}

	void BeginPacket(const CommandRecorder& recorder, uint64_t frame, uint64_t tick) {
		if (m_trace_packet_bounds) {
			m_packet_token = Begin(recorder, frame, tick, 0, nullptr, "submit-envelope");
		}
	}

	void EndPacket(const CommandBuffer& command) {
		if (m_packet_token != 0) {
			command.EndRendering();
			End(command.Handle(), m_packet_token);
			m_packet_token = 0;
		}
	}

	void Save() {
		if (m_pool == nullptr || m_saved) {
			return;
		}
		std::vector<std::array<uint64_t, 2>> queries(m_samples.size() * 2);
		if (!queries.empty()) {
			const auto result = m_graphics.device.getQueryPoolResults(
			    m_pool, 0, static_cast<uint32_t>(queries.size()), queries.size() * sizeof(queries.front()),
			    queries.data(), sizeof(queries.front()),
			    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			if (result != vk::Result::eSuccess && result != vk::Result::eNotReady) {
				Log::WriteToConsoleAndLog("[kyty-pass-timing] cannot read Vulkan timestamps\n");
				return;
			}
		}
		const auto path = m_trigger.parent_path() / "gpu-pass-timing.csv";
		auto temp = path;
		temp += ".tmp";
		std::ofstream out(temp);
		if (!out) {
			Log::WriteToConsoleAndLog("[kyty-pass-timing] cannot open the timing output file\n");
			return;
		}
		const auto period = m_graphics.physical_device_properties.limits.timestampPeriod;
		out << "# timestamp_period_ns=" << period << ",timestamp_mask=" << m_mask << '\n';
		if (m_trace_packet_bounds) {
			out << "# submit-envelope rows enclose GPU passes and are excluded from pass totals\n";
		}
		for (size_t i = 0; i < FrameCount; ++i) {
			out << "# frame=" << m_first_frame + i << ",recorded=" << m_frame_samples[i]
			    << ",dropped=" << m_dropped[i] << '\n';
		}
		out << "renderer_flip,submit_tick,kind,compute_shader,width,height,color_targets,depth,"
		       "start_tick,end_tick,elapsed_ns,pixel_shader,mixed_pixel_shaders,module,dispatch,groups_x,groups_y,groups_z,sample_index\n";
		size_t published = 0;
		for (size_t i = 0; i < m_samples.size(); ++i) {
			const auto [begin, begin_ready] = queries[i * 2];
			const auto [end, end_ready] = queries[i * 2 + 1];
			const auto& sample = m_samples[i];
			if (begin_ready == 0 || end_ready == 0) {
				continue;
			}
			++published;
			out << sample.frame << ',' << sample.tick << ',' << sample.kind
			    << ',' << std::hex << sample.shader << std::dec << ',' << sample.width << ','
			    << sample.height << ',' << sample.color_targets << ',' << sample.depth << ','
			    << (begin & m_mask) << ',' << (end & m_mask) << ','
			    << static_cast<double>((end - begin) & m_mask) * period << ','
			    << std::hex << sample.pixel_shader << std::dec << ',' << sample.mixed_pixel_shaders << ',';
			if (sample.dispatch != nullptr) {
				out << std::hex << sample.module << std::dec << ',' << sample.dispatch << ',';
				if (std::string_view(sample.dispatch).starts_with("indirect")) {
					out << ",,";
				} else {
					out << sample.groups_x << ',' << sample.groups_y << ',' << sample.groups_z;
				}
			} else {
				out << ",,,,";
			}
			out << ',' << i << '\n';
		}
		out.close();
		if (!out) {
			Log::WriteToConsoleAndLog("[kyty-pass-timing] could not finish writing the timing file\n");
			return;
		}
		std::error_code error;
		std::filesystem::rename(temp, path, error);
		if (error) {
			Log::WriteToConsoleAndLog("[kyty-pass-timing] could not publish the timing file\n");
			return;
		}
		m_saved = true;
		m_focus.Save(m_graphics.device, m_trigger.parent_path());
		SaveHostSubmits();
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-pass-timing] wrote {} of {} timestamp pairs to gpu-pass-timing.csv\n",
		    published, m_samples.size()));
	}

private:
	void AdvanceToCombat() {
		m_combat_pending = false;
		m_focus.Close(m_graphics.device);
		m_trigger = m_trigger.parent_path() / "combat-gpu" / m_trigger.filename();
		if (m_pool != nullptr) {
			m_graphics.device.destroyQueryPool(m_pool, nullptr);
			m_pool = nullptr;
		}
		m_armed = false;
		m_saved = false;
		m_save_attempts = 0;
		m_frame_samples.fill(0);
		m_dropped.fill(0);
		m_samples.clear();
		std::scoped_lock lock(m_submit_mutex);
		m_first_submit.store(0, std::memory_order_relaxed);
		m_submits.clear();
	}

	static constexpr size_t   FrameCount = 3;
	static constexpr size_t HostSubmitCapacity = 4096;

	uint32_t QueryCount() const {
		return 2 * m_passes_per_frame * FrameCount;
	}

	struct HostSubmit {
		uint64_t frame = 0;
		int64_t begin = 0;
		int64_t publish = 0;
		int64_t record_end = 0;
		int64_t copies_ready = 0;
		int64_t queue_request = 0;
		int64_t queue_lock = 0;
		int64_t queue_return = 0;
		uint32_t waits = 0;
		bool has_buffer = false;
	};

	HostSubmit* FindHostSubmit(uint64_t tick) {
		const auto first = m_first_submit.load(std::memory_order_relaxed);
		const auto index = tick - first;
		return first != 0 && index < m_submits.size() && m_submits[index].begin != 0
		           ? &m_submits[index] : nullptr;
	}

	void SaveHostSubmits() {
		if (!m_trace_submits) {
			return;
		}
		std::vector<HostSubmit> submits;
		uint64_t first;
		{
			std::scoped_lock lock(m_submit_mutex);
			first = m_first_submit.exchange(0, std::memory_order_relaxed);
			submits.swap(m_submits);
		}
		const auto path = m_trigger.parent_path() / "gpu-submit-host.csv";
		auto temp = path;
		temp += ".tmp";
		std::ofstream out(temp);
		out << "renderer_flip,submit_tick,begin_ns,publish_ns,record_end_ns,host_copies_ready_ns,"
		       "queue_request_ns,queue_lock_ns,queue_return_ns,wait_semaphores,has_command_buffer\n";
		for (size_t i = 0; i < submits.size(); ++i) {
			const auto& s = submits[i];
			if (s.begin != 0) {
				out << s.frame << ',' << first + i << ',' << s.begin << ',' << s.publish << ','
				    << s.record_end << ',' << s.copies_ready << ',' << s.queue_request << ','
				    << s.queue_lock << ',' << s.queue_return << ',' << s.waits << ',' << s.has_buffer << '\n';
			}
		}
		out.close();
		std::error_code error;
		if (out) {
			std::filesystem::rename(temp, path, error);
		}
		if (!out || error) {
			Log::WriteToConsoleAndLog("[kyty-pass-timing] could not publish the host submit timing file\n");
		}
	}

	struct Sample {
		uint64_t frame;
		uint64_t tick;
		uint64_t shader;
		const char* kind;
		uint32_t width;
		uint32_t height;
		uint32_t color_targets;
		bool     depth;
		uint64_t pixel_shader;
		bool     mixed_pixel_shaders;
		uint64_t module = 0;
		const char* dispatch = nullptr;
		uint32_t groups_x = 0;
		uint32_t groups_y = 0;
		uint32_t groups_z = 0;
	};

	FocusGraphicsTiming                   m_focus;
	GraphicContext&                       m_graphics;
	std::filesystem::path                 m_trigger;
	std::chrono::steady_clock::time_point m_next_poll {};
	bool                                   m_armed = false;
	bool                                   m_saved = false;
	uint32_t                               m_save_attempts = 0;
	vk::QueryPool                          m_pool = nullptr;
	uint64_t                               m_mask = 0;
	uint64_t                               m_first_frame = 0;
	std::array<uint32_t, FrameCount>        m_frame_samples {};
	std::array<uint32_t, FrameCount>        m_dropped {};
	std::vector<Sample>                    m_samples;
	const bool                             m_trace_submits;
	const uint32_t                         m_passes_per_frame;
	const bool                             m_trace_packet_bounds;
	bool                                   m_combat_pending;
	uint32_t                               m_packet_token = 0;
	std::mutex                             m_submit_mutex;
	std::atomic<uint64_t>                  m_first_submit {0};
	std::vector<HostSubmit>                m_submits;
};
#endif

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	return m_buffers[Acquire(m_master.CurrentTick())];
}

std::pair<vk::CommandBuffer, size_t> CommandScheduler::CommandPool::CommitDeferred() {
	const auto index = Acquire(std::numeric_limits<uint64_t>::max());
	return {m_buffers[index], index};
}

void CommandScheduler::CommandPool::Retire(size_t index, uint64_t tick) {
	EXIT_IF(index >= m_ticks.size() || m_ticks[index] != std::numeric_limits<uint64_t>::max());
	m_ticks[index] = tick;
}

size_t CommandScheduler::CommandPool::Acquire(uint64_t busy_until) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick, busy_until](size_t begin,
	                                                  size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = busy_until;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = busy_until;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return *found;
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics,
                                   bool deferred_recording)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_deferred(deferred_recording),
      m_stream(&m_chunk_queue), m_command(*this, deferred_recording ? &m_stream : nullptr),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
#if defined(ANYPS5_KYTY)
	if (const auto* trigger = std::getenv("KYTY_GPU_PASS_TIMING_TRIGGER"); trigger && *trigger) {
		const auto* trace_submits = std::getenv("KYTY_GPU_SUBMIT_TIMING");
		const auto* trace_packet_bounds = std::getenv("KYTY_GPU_PACKET_TIMING");
		const auto* capture_combat = std::getenv("KYTY_GPU_COMBAT_TIMING");
		m_pass_timing = std::make_unique<PassTiming>(
		    graphics, trigger, trace_submits != nullptr && std::string_view(trace_submits) == "1",
		    trace_packet_bounds != nullptr && std::string_view(trace_packet_bounds) == "1",
		    capture_combat != nullptr && std::string_view(capture_combat) == "1");
	}
#endif
	if (m_deferred) {
		m_host_copies = std::make_unique<HostCopyQueue>(
		    [] { Config::ConfigureGpuStageThread(Config::GpuStageThread::HostCopy); });
		m_recording_thread = std::jthread([this] { RecordingThread(); });
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		Common::UniqueLock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock.Native(), [this]() KYTY_REQUIRES(m_operation_mutex) { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	if (m_recording_thread.joinable()) {
		m_chunk_queue.Stop();
		m_recording_thread.join();
	}
#if defined(ANYPS5_KYTY)
	if (m_pass_timing) {
		m_pass_timing->Save();
		m_pass_timing.reset();
	}
#endif
	PopPendingOperations();
	DrainPriorityOperations();
#if defined(ANYPS5_KYTY)
	DrainFlipCompletions();
	if (m_flip_thread.joinable()) {
		m_flip_thread.request_stop();
		m_flip_thread.join();
	}
#endif
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		Common::ScopedLock lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(const HW::Context& registers, const HW::UserConfig& user_config,
                             const HW::Shader& shaders) {
	if (!m_command.IsInvalid()) {
		m_command.Bind(registers, user_config, shaders);
		return;
	}
	{
		Common::ScopedLock lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);
	BeginNext();
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	bool refreshed = false;
	if (const auto front = m_pending_front_tick.load(std::memory_order_acquire);
	    front == UINT64_MAX) {
		return;
	} else if (!m_master.IsFree(front)) {
		constexpr uint32_t RefreshInterval = 8;
		if (const auto skips = m_pending_skips.load(std::memory_order_relaxed) + 1;
		    skips < RefreshInterval) {
			m_pending_skips.store(skips, std::memory_order_relaxed);
			return;
		}
		m_pending_skips.store(0, std::memory_order_relaxed);
		m_master.Refresh();
		refreshed = true;
		if (!m_master.IsFree(front)) {
			return;
		}
	}
	for (;;) {
		PendingOperation operation;
		{
			Common::ScopedLock lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			if (!m_master.IsFree(m_pending_operations.front().tick)) {
				if (refreshed) {
					return;
				}
				m_master.Refresh();
				refreshed = true;
				if (!m_master.IsFree(m_pending_operations.front().tick)) {
					return;
				}
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
			m_pending_front_tick.store(m_pending_operations.empty()
			                               ? UINT64_MAX
			                               : m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	QueueOperation(TrackGuestOperation(std::move(operation), address, size));
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                              uint64_t address, uint64_t size) {
	QueuePriorityOperation(TrackGuestOperation(std::move(operation), address, size));
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&&                     operation,
                                              const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	QueuePriorityOperation(TrackGuestOperation(std::move(operation), ranges));
}

#if defined(ANYPS5_KYTY)
void CommandScheduler::DeferPriorityOperationAfter(uint64_t tick, Common::UniqueFunction<void>&& operation,
                                                   uint64_t address, uint64_t size) {
	CheckActive();
	EXIT_IF(tick == 0 || tick > CurrentTick());
	auto tracked = TrackGuestOperation(std::move(operation), address, size);
	{
		Common::UniqueLock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Open) {
			if (!m_priority_operations.empty()) {
				tick = std::max(tick, m_priority_operations.back().tick);
			}
			tick = std::max(tick, m_priority_active_tick);
			m_priority_operations.push({std::move(tracked), tick});
			lock.unlock();
			m_operation_available.notify_one();
			return;
		}
	}
	QueuePriorityOperation(std::move(tracked));
}
#endif

void CommandScheduler::DeferHostOperation(Common::UniqueFunction<void>&& operation,
                                          bool                           priority) {
	if (priority) {
		QueuePriorityOperation(std::move(operation));
	} else {
		QueueOperation(std::move(operation));
	}
}

#if defined(ANYPS5_KYTY)
static bool EarlyFlipEnabled() {
	static const bool enabled = [] {
		const char*            value = std::getenv("KYTY_EARLY_FLIP");
		const std::string_view text {value == nullptr ? "" : value};
		if (!text.empty() && text != "0" && text != "1") {
			EXIT("invalid KYTY_EARLY_FLIP: %s (0 or 1)\n", value);
		}
		return text != "0";
	}();
	return enabled;
}

void CommandScheduler::DeferFlipCompletion(Common::UniqueFunction<void>&& complete) {
	CheckActive();
	EXIT_IF(!complete);
	const uint64_t previous = std::exchange(m_last_flip_tick, CurrentTick());
	if (!m_flips_in_place && InPlaceMemory::Live() && previous < CurrentTick()) {
		if (EarlyFlipEnabled()) {
			DrainFlipCompletions();
		}
		m_flips_in_place = true;
	}
	if (m_flips_in_place && previous != 0 && previous < CurrentTick()) {
		const auto started = std::chrono::steady_clock::now();
		m_master.Wait(previous);
		InPlaceMemory::NoteBoundedFlip(static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
	}
	if (!EarlyFlipEnabled() || m_flips_in_place) {
		QueuePriorityOperation(std::move(complete));
		return;
	}
	std::call_once(m_flip_thread_once, [this] {
		m_flip_thread =
		    std::jthread([this](std::stop_token stop) { FlipCompletionThread(stop); });
	});
	Common::ScopedLock lock(m_flip_mutex);
	m_flip_completions.push_back({std::move(complete), m_last_flip_tick, previous});
	m_flips_pending.store(true, std::memory_order_release);
}

void CommandScheduler::FlipCompletionThread(std::stop_token stop) {
	for (;;) {
		FlipCompletion flip;
		{
			Common::UniqueLock lock(m_flip_mutex);
			if (!m_flip_changed.wait(lock, stop, [this]() KYTY_REQUIRES(m_flip_mutex) {
				    return !m_flip_completions.empty() &&
				           SubmittedTick() >= m_flip_completions.front().submitted_tick;
			    })) {
				return;
			}
			flip = std::move(m_flip_completions.front());
			m_flip_completions.pop_front();
			m_flips_pending.store(!m_flip_completions.empty(), std::memory_order_release);
			m_flip_running = true;
		}
		if (flip.previous_tick != 0) {
			m_master.Wait(flip.previous_tick);
			WaitPriorityOperations(flip.previous_tick);
		}
		RunOperation(std::move(flip.complete));
		{
			Common::ScopedLock lock(m_flip_mutex);
			m_flip_running = false;
		}
		m_flip_changed.notify_all();
	}
}

void CommandScheduler::DrainFlipCompletions() {
	Common::UniqueLock lock(m_flip_mutex);
	m_flip_changed.wait(lock, [this]() KYTY_REQUIRES(m_flip_mutex) { return m_flip_completions.empty() && !m_flip_running; });
}
#endif

Common::UniqueFunction<void>
CommandScheduler::TrackGuestOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	EXIT_IF(!operation);
	uint64_t id = 0;
	{
		Common::ScopedLock lock(m_guest_writes_mutex);
		id = ++m_next_guest_write;
		m_guest_writes.push_back(
		    {id, address, size > UINT64_MAX - address ? UINT64_MAX : address + size});
	}
	return [this, id, operation = std::move(operation)]() mutable {
		operation();
		Common::ScopedLock lock(m_guest_writes_mutex);
		const auto it = std::find_if(m_guest_writes.begin(), m_guest_writes.end(),
		                             [id](const GuestWrite& write) { return write.id == id; });
		EXIT_IF(it == m_guest_writes.end());
		m_guest_writes.erase(it);
	};
}

Common::UniqueFunction<void>
CommandScheduler::TrackGuestOperation(Common::UniqueFunction<void>&&                     operation,
                                      const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	EXIT_IF(!operation);
	uint64_t id = 0;
	{
		Common::ScopedLock lock(m_guest_writes_mutex);
		id = ++m_next_guest_write;
		for (const auto& [address, size]: ranges) {
			m_guest_writes.push_back(
			    {id, address, size > UINT64_MAX - address ? UINT64_MAX : address + size});
		}
	}
	return [this, id, operation = std::move(operation)]() mutable {
		operation();
		Common::ScopedLock lock(m_guest_writes_mutex);
		std::erase_if(m_guest_writes, [id](const GuestWrite& write) { return write.id == id; });
	};
}

#if defined(ANYPS5_KYTY)
uint64_t CommandScheduler::NewestPendingGuestOperation(uint64_t address, uint64_t size,
                                                        uint64_t after) {
	const auto      end = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	Common::ScopedLock lock(m_guest_writes_mutex);
	for (auto it = m_guest_writes.rbegin(); it != m_guest_writes.rend() && it->id > after; ++it) {
		if (it->begin < end && address < it->end) {
			return it->id;
		}
	}
	return 0;
}
#endif

bool CommandScheduler::HasPendingGuestOperations(uint64_t address, uint64_t size) {
	const auto      end = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	Common::ScopedLock lock(m_guest_writes_mutex);
	return std::any_of(m_guest_writes.begin(), m_guest_writes.end(), [&](const GuestWrite& write) {
		return write.begin < end && address < write.end;
	});
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	Common::UniqueLock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		if (m_pending_operations.size() == 1) {
			m_pending_front_tick.store(m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock.Native(),
	                           [this]() KYTY_REQUIRES(m_operation_mutex) { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::QueuePriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	Common::UniqueLock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock.Native(),
	                           [this]() KYTY_REQUIRES(m_operation_mutex) { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	Common::ScopedLock lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			Common::UniqueLock lock(m_operation_mutex);
			m_operation_available.wait(lock.Native(), [this, &stop]() KYTY_REQUIRES(m_operation_mutex) {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			Common::ScopedLock lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	Common::UniqueLock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock.Native(), [this]() KYTY_REQUIRES(m_operation_mutex) { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}


void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
#if defined(ANYPS5_KYTY)
	if (tick >= CurrentTick() && Active() && !m_command.IsInvalid() && GuestGpu::IsGpuThread() &&
	    HasPendingPriorityOperations()) {
		m_command.EndRendering();
		Flush();
	}
#endif
	Common::UniqueLock lock(m_operation_mutex);
	m_operation_available.wait(lock.Native(), [this, tick]() KYTY_REQUIRES(m_operation_mutex) {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
#if defined(ANYPS5_KYTY)
	if (m_pass_timing && this == &m_context.GetCommandScheduler()) {
		m_pass_timing->Poll(m_context.FlipNumber(), [this] {
			m_master.Refresh();
			return m_master.KnownGpuTick();
		});
		m_pass_timing->BeginHostSubmit(m_context.FlipNumber(), CurrentTick());
	}
#endif
	if (m_deferred) {
		m_command.m_open = true;
	} else {
		m_command.m_buffer = m_command_pool.Commit();
	}
	m_command.Begin();
#if defined(ANYPS5_KYTY)
	QueueCheckpoints::Poll(m_graphics.device, m_context.FlipNumber());
	QueueCheckpoints::Mark(m_graphics.device, m_command.Handle(), m_context.FlipNumber(), CurrentTick(), "packet-begin");
	if (m_pass_timing) {
		m_pass_timing->BeginPacket(m_command.Handle(), m_context.FlipNumber(), CurrentTick());
	}
#endif
	m_dynamic_state.Invalidate();
	return m_command;
}

#if defined(ANYPS5_KYTY)
uint32_t CommandScheduler::BeginGraphicsTiming(const RenderState& state, uint64_t pixel_shader) {
	QueueCheckpoints::Mark(m_graphics.device, m_command.Handle(), m_context.FlipNumber(), CurrentTick(), "graphics", pixel_shader);
	return m_pass_timing ? m_pass_timing->Begin(m_command.Handle(), m_context.FlipNumber(),
	                                            CurrentTick(), 0, &state, "graphics", pixel_shader)
	                     : 0;
}

void CommandScheduler::NoteGraphicsTimingShader(uint32_t token, uint64_t pixel_shader) {
	if (token != 0) {
		m_pass_timing->NoteGraphicsShader(token, pixel_shader);
	}
}

uint32_t CommandScheduler::BeginComputeTiming(uint64_t shader, uint64_t module, const char* dispatch,
                                             uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
#if defined(_WIN32)
	if (FrameCapture::Get().Enabled()) {
		FrameCapture::Get().Record({2, m_context.FlipNumber(), CurrentTick(), shader, module, groups_x,
		    groups_y, groups_z, static_cast<uint64_t>(std::strcmp(dispatch, "indirect") == 0)});
	}
#endif
	QueueCheckpoints::Mark(m_graphics.device, m_command.Handle(), m_context.FlipNumber(), CurrentTick(), "compute", shader, module);
	const auto token = m_pass_timing ? m_pass_timing->Begin(m_command.Handle(), m_context.FlipNumber(), CurrentTick(), shader, nullptr, "compute") : 0;
	if (token != 0) {
		m_pass_timing->NoteComputeDispatch(token, module, dispatch, groups_x, groups_y, groups_z);
	}
	return token;
}

uint32_t CommandScheduler::BeginPassTiming(uint64_t shader, const char* kind) {
	QueueCheckpoints::Mark(m_graphics.device, m_command.Handle(), m_context.FlipNumber(), CurrentTick(), kind, shader);
	return m_pass_timing ? m_pass_timing->Begin(m_command.Handle(), m_context.FlipNumber(),
	                                            CurrentTick(), shader, nullptr, kind)
	                     : 0;
}

void CommandScheduler::EndPassTiming(uint32_t token) {
	if (token != 0) {
		m_pass_timing->End(m_command.Handle(), token);
	}
}
#endif

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	if (m_submit_hook != nullptr) {
		m_submit_hook(m_submit_hook_data, m_command);
	}
	if (InPlaceMemory::Live()) {
		m_command.EndRendering();
		vk::MemoryBarrier host {};
		host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		host.dstAccessMask = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
		m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {},
		                                   1, &host, 0, nullptr, 0, nullptr);
	}

#if defined(ANYPS5_KYTY)
	if (m_pass_timing) {
		m_pass_timing->EndPacket(m_command);
	}
#endif
	#if defined(ANYPS5_KYTY)
	QueueCheckpoints::Mark(m_graphics.device, m_command.Handle(), m_context.FlipNumber(), CurrentTick(), "packet-end");
#endif
	m_command.End();
	const auto     tick = m_master.NextTick();
	if (m_deferred) {
		auto* recorded = m_stream.RecordSpecial<RecordedSubmit>(CommandStream::RecordKind::Submit);
		recorded->info = submit;
		recorded->tick = tick;
		recorded->host_copies = m_host_copies != nullptr ? m_host_copies->Flush() : 0;
#if defined(ANYPS5_KYTY)
		if (m_pass_timing && m_pass_timing->MaybeTraceHostSubmit(tick)) {
			m_pass_timing->PublishHostSubmit(tick, PassTiming::HostTime());
		}
#endif
		m_stream.Publish();
		m_command.m_open = false;
		return tick;
	}

#if defined(ANYPS5_KYTY)
	if (m_pass_timing && m_pass_timing->MaybeTraceHostSubmit(tick)) {
		m_pass_timing->PublishHostSubmit(tick, PassTiming::HostTime());
	}
#endif
	QueueSubmit(m_command.m_buffer, submit, tick);
	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick) {
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);
	submit.AddSignal(m_master.Handle(), tick);

	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = buffer != nullptr ? 1 : 0;
	submit_info.pCommandBuffers      = buffer != nullptr ? &buffer : nullptr;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	vk::Result result;
#if defined(ANYPS5_KYTY)
	const bool trace = m_pass_timing && m_pass_timing->MaybeTraceHostSubmit(tick);
	const auto queue_request = trace ? PassTiming::HostTime() : 0;
	int64_t queue_lock = 0;
	int64_t queue_return = 0;
#endif
	{
		Common::LockGuard lock(graphics.queue_mutex);
#if defined(ANYPS5_KYTY)
		if (trace) {
			queue_lock = PassTiming::HostTime();
		}
#endif
		result = graphics.queue.submit(1, &submit_info, nullptr);
#if defined(ANYPS5_KYTY)
		if (trace) {
			queue_return = PassTiming::HostTime();
		}
#endif
	}
#if defined(ANYPS5_KYTY)
	if (trace) {
		m_pass_timing->QueueHostSubmit(tick, queue_request, queue_lock, queue_return,
		                               submit.num_wait_semaphores, buffer != nullptr);
	}
#endif
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}
	m_submitted_tick.store(tick, std::memory_order_release);
#if defined(ANYPS5_KYTY)
	if (m_flips_pending.load(std::memory_order_acquire)) {
		{
			Common::ScopedLock lock(m_flip_mutex);
		}
		m_flip_changed.notify_all();
	}
#endif
}

void CommandScheduler::RecordingThread() {
	KYTY_PROFILER_THREAD("Thread_GpuRecord");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Recording);
	vk::CommandBuffer buffer       = nullptr;
	size_t            buffer_index = 0;
	const auto        acquire      = [this, &buffer, &buffer_index] {
		if (buffer == nullptr) {
			std::tie(buffer, buffer_index) = m_command_pool.CommitDeferred();
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(buffer.begin(&begin_info) != vk::Result::eSuccess);
		}
		return buffer;
	};
	for (;;) {
		auto chunk = m_chunk_queue.Pop();
		if (chunk == nullptr) {
			break;
		}
		CommandStream::Replay(*chunk, acquire, [&](const CommandStream::Entry& entry) {
			EXIT_IF(entry.kind != CommandStream::RecordKind::Submit);
			SubmitRecorded(buffer, buffer_index,
			               *static_cast<const RecordedSubmit*>(entry.Payload()));
		});
		m_chunk_queue.Recycle(std::move(chunk));
	}
	EXIT_IF(buffer != nullptr);
}

void CommandScheduler::SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
                                      const RecordedSubmit& recorded) {
	if (buffer != nullptr) {
		EXIT_NOT_IMPLEMENTED(buffer.end() != vk::Result::eSuccess);
	}
#if defined(ANYPS5_KYTY)
	const bool trace = m_pass_timing && m_pass_timing->MaybeTraceHostSubmit(recorded.tick);
	const auto record_end = trace ? PassTiming::HostTime() : 0;
#endif
	if (m_host_copies != nullptr) {
		m_host_copies->WaitFor(recorded.host_copies);
	}
#if defined(ANYPS5_KYTY)
	if (trace) {
		m_pass_timing->ReadyHostSubmit(recorded.tick, record_end, PassTiming::HostTime());
	}
#endif
	auto submit = recorded.info;
	QueueSubmit(buffer, submit, recorded.tick);
	if (buffer != nullptr) {
		m_command_pool.Retire(buffer_index, recorded.tick);
		buffer = nullptr;
	}
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

}
