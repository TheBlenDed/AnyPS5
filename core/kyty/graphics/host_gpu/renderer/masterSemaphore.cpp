#include "graphics/host_gpu/renderer/masterSemaphore.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/queueCheckpoints.h"
#endif


#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

#if defined(ANYPS5_KYTY)
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#endif

namespace Libs::Graphics {

#if defined(ANYPS5_KYTY)
void MasterSemaphore::ReportFailure(const char* operation, vk::Result result, uint64_t wait_tick) const {
	QueueCheckpoints::Report(m_graphics.device, m_graphics.queue, "main", result);
	if (m_graphics.readback_queue != m_graphics.queue) {
		QueueCheckpoints::Report(m_graphics.device, m_graphics.readback_queue, "readback", result);
	}
	const auto known = KnownGpuTick();
	const auto next = m_work_next.load(std::memory_order_relaxed);
	std::fprintf(stderr, "[kyty-timeline-error] operation=%s result=%d wait_tick=%" PRIu64
	             " known_tick=%" PRIu64 " next_tick=%" PRIu64 "\n",
	             operation, static_cast<int>(result), wait_tick, known, CurrentTick());
	std::array<std::pair<uint64_t, uint64_t>, 24> recent {};
	size_t count = 0;
	for (size_t i = 1; i <= m_work.size() && count < recent.size(); ++i) {
		const auto& slot = m_work[(next - i) % m_work.size()];
		const auto tick = slot.tick.load(std::memory_order_relaxed);
		const auto shader = slot.shader.load(std::memory_order_relaxed);
		if (tick <= known || shader == 0) {
			continue;
		}
		const std::pair entry {tick, shader};
		if (std::find(recent.begin(), recent.begin() + count, entry) == recent.begin() + count) {
			recent[count++] = entry;
			std::fprintf(stderr, "[kyty-timeline-work] recent_tick=%" PRIu64 " compute=%016" PRIx64 "\n", tick, shader);
		}
	}
	std::fflush(stderr);
}

void MasterSemaphore::ReportStall(uint64_t tick) {
	static std::atomic_flag reported_gpu     = ATOMIC_FLAG_INIT;
	static std::atomic_flag reported_pending = ATOMIC_FLAG_INIT;
	const bool              pending          = tick >= CurrentTick();
	if ((pending ? reported_pending : reported_gpu).test_and_set()) {
		return;
	}
	uint64_t reached = 0;
	(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &reached);
	if (pending) {
		std::fprintf(stderr,
		             "[kyty] tick %" PRIu64 " has been awaited for 10 s and is not submitted yet (next tick %" PRIu64
		             ", the GPU is at %" PRIu64 "): the CPU side holds it\n",
		             tick, CurrentTick(), reached);
		return;
	}
	std::fprintf(stderr,
	             "[kyty] the GPU has not reached tick %" PRIu64 " in 10 s (it is at %" PRIu64 ", %" PRIu64
	             " ticks issued after it): the GPU stopped, look for an NVIDIA Xid in dmesg\n",
	             tick, reached, CurrentTick() - 1 - reached);
	std::array<uint64_t, 24> shaders {};
	size_t                   count = 0;
	const auto               next  = m_work_next.load(std::memory_order_relaxed);
	for (uint32_t i = 1; i <= m_work.size() && count < shaders.size(); i++) {
		const auto& work = m_work[(next - i) % m_work.size()];
		if (work.tick.load(std::memory_order_relaxed) != reached + 1) {
			continue;
		}
		const auto shader = work.shader.load(std::memory_order_relaxed);
		if (std::find(shaders.begin(), shaders.begin() + count, shader) == shaders.begin() + count) {
			shaders[count++] = shader;
		}
	}
	std::fprintf(stderr, "[kyty] compute shaders of the unfinished tick %" PRIu64 ", last first:", reached + 1);
	for (size_t i = 0; i < count; i++) {
		std::fprintf(stderr, " 0x%016" PRIx64, shaders[i]);
	}
	std::fprintf(stderr, count == 0 ? " none recorded\n" : "\n");
}
#endif

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
#if defined(ANYPS5_KYTY)
	if (result != vk::Result::eSuccess) {
		ReportFailure("getSemaphoreCounterValue", result, 0);
	}
#endif
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
#if defined(ANYPS5_KYTY)
	constexpr uint64_t StallNs = 10'000'000'000ull;
	auto result = m_graphics.device.waitSemaphores(&wait_info, StallNs);
	while (result == vk::Result::eTimeout) {
		ReportStall(tick);
		result = m_graphics.device.waitSemaphores(&wait_info, StallNs);
	}
	if (result != vk::Result::eSuccess) {
		ReportFailure("waitSemaphores", result, tick);
	}
#else
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
#endif
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

}
