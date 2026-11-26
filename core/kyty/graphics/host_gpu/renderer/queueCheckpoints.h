#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>

namespace Libs::Graphics::QueueCheckpoints {

inline bool Requested() {
	static const bool requested = [] {
		const auto* value = std::getenv("KYTY_GPU_CHECKPOINTS");
		return value && std::strcmp(value, "1") == 0;
	}();
	return requested;
}

struct Entry {
	uintptr_t token;
	uint64_t frame;
	uint64_t tick;
	const char* operation;
	uint64_t shader;
	uint64_t detail;
};

struct History {
	std::mutex mutex;
	std::array<Entry, 65536> entries {};
	uintptr_t next = 1;
};

inline History& Records() {
	static History history;
	return history;
}

inline std::atomic<VkDevice> armed {};

inline void Poll(vk::Device device, uint64_t frame) {
	if (!Requested() || armed.load(std::memory_order_acquire) == static_cast<VkDevice>(device)) return;
	static thread_local uint64_t previous_frame = UINT64_MAX;
	if (previous_frame == frame) return;
	previous_frame = frame;
	const char* trigger = std::getenv("KYTY_GPU_CHECKPOINT_TRIGGER");
	std::error_code error;
	if (!trigger || !std::filesystem::is_regular_file(trigger, error) ||
	    !device.getProcAddr("vkCmdSetCheckpointNV") || !device.getProcAddr("vkGetQueueCheckpointDataNV")) return;
	armed.store(device, std::memory_order_release);
	std::fprintf(stderr, "[kyty-checkpoints] armed frame=%" PRIu64 "\n", frame);
}

template <class Recorder>
void Mark(vk::Device device, const Recorder& recorder, uint64_t frame, uint64_t tick,
          const char* operation, uint64_t shader = 0, uint64_t detail = 0) {
	if (armed.load(std::memory_order_acquire) != static_cast<VkDevice>(device)) return;
	auto& history = Records();
	uintptr_t token;
	{
		std::lock_guard lock(history.mutex);
		token = history.next++;
		if (token == 0) token = history.next++;
		history.entries[token % history.entries.size()] = {token, frame, tick, operation, shader, detail};
	}
	recorder.setCheckpointNV(reinterpret_cast<const void*>(token));
}

inline void Describe(const char* queue, uint32_t stage, uintptr_t token) {
	auto& history = Records();
	std::lock_guard lock(history.mutex);
	const auto& entry = history.entries[token % history.entries.size()];
	if (!token || entry.token != token) {
		std::fprintf(stderr, "[kyty-checkpoint] queue=%s stage=0x%x token=%" PRIuPTR " unavailable\n", queue, stage, token);
		return;
	}
	std::fprintf(stderr, "[kyty-checkpoint] queue=%s stage=0x%x token=%" PRIuPTR " frame=%" PRIu64
	             " tick=%" PRIu64 " operation=%s shader=%016" PRIx64 " detail=%016" PRIx64 "\n",
	             queue, stage, token, entry.frame, entry.tick, entry.operation, entry.shader, entry.detail);
}

inline void Report(vk::Device device, vk::Queue queue, const char* name, vk::Result result) {
	if (result != vk::Result::eErrorDeviceLost || !queue ||
	    armed.load(std::memory_order_acquire) != static_cast<VkDevice>(device)) return;
	std::array<vk::CheckpointDataNV, 32> data {};
	uint32_t count = static_cast<uint32_t>(data.size());
	queue.getCheckpointDataNV(&count, data.data());
	std::fprintf(stderr, "[kyty-checkpoints] queue=%s reported=%u\n", name, count);
	for (uint32_t i = 0; i < count; ++i) {
		Describe(name, static_cast<uint32_t>(data[i].stage), reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker));
	}
}

}
