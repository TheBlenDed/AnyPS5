#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERREADBACKWINDOWS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERREADBACKWINDOWS_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

class EagerReadbackWindows final {
public:
	static constexpr uint64_t WindowSize    = 64 * 1024;
	static constexpr size_t   Slots         = 256;
	static constexpr size_t   Probes        = 4;
	static constexpr uint32_t HotHits       = 2;
	static constexpr uint64_t HitFrames     = 16;
	static constexpr uint32_t IdleLimit     = 8;
	static constexpr uint32_t WakeHits      = 1;
	static constexpr uint64_t HotFrames     = 64;
	static constexpr uint64_t BlockedFrames = 4096;
	static constexpr uint64_t ScanWindows   = 8;
	static constexpr size_t   None          = SIZE_MAX;

	enum class Fault : uint8_t { Cold, Hot, Woken, Asleep, Blocked, Dropped };
	enum class Published : uint8_t { Touched, Idle, Asleep };

	struct Settings {
		uint32_t hot_hits   = HotHits;
		uint64_t hit_frames = HitFrames;
		uint32_t idle_limit = IdleLimit;
		uint32_t wake_hits  = WakeHits;
		size_t   probes     = Probes;
	};

	struct Window {
		std::atomic<uint64_t> key {0};
		std::atomic<uint8_t>  touched {0};
		uint64_t              last_frame = 0;
		uint64_t              hit_frame  = 0;
		uint32_t              hits       = 0;
		uint32_t              idle       = 0;
		uint32_t              wake       = 0;
		bool                  last_idle  = false;
		bool                  asleep     = false;
		bool                  blocked    = false;
		bool                  queued     = false;
		bool                  in_flight  = false;
		bool                  again      = false;
	};

	EagerReadbackWindows() = default;
	explicit EagerReadbackWindows(const Settings& settings) noexcept: m_settings(settings) {
		m_settings.hot_hits   = std::max<uint32_t>(settings.hot_hits, 1);
		m_settings.idle_limit = std::max<uint32_t>(settings.idle_limit, 1);
		m_settings.wake_hits  = std::max<uint32_t>(settings.wake_hits, 1);
		m_settings.probes     = std::clamp<size_t>(settings.probes, 1, Slots);
	}
	EagerReadbackWindows(const EagerReadbackWindows&)            = delete;
	EagerReadbackWindows& operator=(const EagerReadbackWindows&) = delete;

	[[nodiscard]] static constexpr uint64_t Start(uint64_t address) noexcept { return address & ~(WindowSize - 1); }
	[[nodiscard]] static constexpr size_t   Home(uint64_t address) noexcept {
		return static_cast<size_t>(((address / WindowSize) * 0x9E3779B97F4A7C15ull) >> 56) % Slots;
	}

	[[nodiscard]] const Settings& Parameters() const noexcept { return m_settings; }
	[[nodiscard]] Window&       At(size_t index) noexcept { return m_windows[index]; }
	[[nodiscard]] const Window& At(size_t index) const noexcept { return m_windows[index]; }
	[[nodiscard]] uint64_t      Begin(size_t index) const noexcept {
		return m_windows[index].key.load(std::memory_order_relaxed) - 1;
	}

	[[nodiscard]] size_t Find(uint64_t address) const noexcept {
		const auto key  = Start(address) + 1;
		const auto home = Home(address);
		for (size_t probe = 0; probe < m_settings.probes; probe++) {
			const auto index = (home + probe) % Slots;
			if (m_windows[index].key.load(std::memory_order_relaxed) == key) {
				return index;
			}
		}
		return None;
	}

	[[nodiscard]] bool Active(size_t index) const noexcept {
		const auto& window = m_windows[index];
		return window.key.load(std::memory_order_relaxed) != 0 && window.hits >= m_settings.hot_hits && !window.asleep && !window.blocked;
	}

	[[nodiscard]] size_t ActiveCount() const noexcept {
		size_t count = 0;
		for (size_t index = 0; index < Slots; index++) {
			count += Active(index);
		}
		return count;
	}

	Fault NoteFault(uint64_t address, uint64_t frame, size_t& index) {
		index = Find(address);
		if (index == None) {
			index = Insert(address, frame);
			if (index == None) {
				return Fault::Dropped;
			}
		}
		auto& window = m_windows[index];
		if (window.blocked) {
			window.last_frame = frame;
			return Fault::Blocked;
		}
		window.touched.store(1, std::memory_order_relaxed);
		if (window.asleep) {
			if (window.wake != 0 && frame - std::min(frame, window.hit_frame) > m_settings.hit_frames) {
				window.wake = 0;
			}
			if (window.wake == 0) {
				window.hit_frame = frame;
			}
			window.last_frame = frame;
			if (++window.wake < m_settings.wake_hits) {
				return Fault::Asleep;
			}
			window.asleep = false;
			window.wake   = 0;
			window.idle   = 0;
			window.hits   = m_settings.hot_hits;
			return Fault::Woken;
		}
		if (window.hits < m_settings.hot_hits) {
			if (window.hits != 0 && frame - std::min(frame, window.hit_frame) > m_settings.hit_frames) {
				window.hits = 0;
			}
			if (window.hits == 0) {
				window.hit_frame = frame;
			}
			window.hits++;
		}
		window.idle       = 0;
		window.last_frame = frame;
		return window.hits >= m_settings.hot_hits ? Fault::Hot : Fault::Cold;
	}

	[[nodiscard]] bool Full(uint64_t frame) const noexcept {
		for (const auto& window: m_windows) {
			if (window.key.load(std::memory_order_relaxed) == 0 || Evictable(window, frame)) {
				return false;
			}
		}
		return true;
	}

	size_t Block(uint64_t begin, uint64_t end, uint64_t frame) {
		size_t blocked = 0;
		for (auto address = Start(begin); address < end; address += WindowSize) {
			auto index = Find(address);
			if (index == None) {
				index = Insert(address, frame);
				if (index == None) {
					continue;
				}
			}
			auto& window = m_windows[index];
			blocked            += !window.blocked;
			window.blocked     = true;
			window.asleep      = false;
			window.hits        = 0;
			window.idle        = 0;
			window.wake        = 0;
			window.last_frame  = frame;
		}
		return blocked;
	}

	void NoteGuestWrite(uint64_t address) noexcept {
		const auto key  = Start(address) + 1;
		const auto home = Home(address);
		for (size_t probe = 0; probe < m_settings.probes; probe++) {
			auto& window = m_windows[(home + probe) % Slots];
			if (window.key.load(std::memory_order_relaxed) == key) {
				if (window.touched.load(std::memory_order_relaxed) == 0) {
					window.touched.store(1, std::memory_order_relaxed);
				}
				return;
			}
		}
	}

	Published NotePublished(size_t index, uint64_t frame) {
		auto& window = m_windows[index];
		if (window.touched.exchange(0, std::memory_order_relaxed) != 0) {
			window.idle       = 0;
			window.last_idle  = false;
			window.last_frame = frame;
			return Published::Touched;
		}
		window.last_idle = true;
		if (window.asleep || window.blocked || ++window.idle < m_settings.idle_limit) {
			return Published::Idle;
		}
		window.asleep = true;
		window.idle   = 0;
		window.wake   = 0;
		return Published::Asleep;
	}

	template <typename Func>
	void ForEachActiveIn(uint64_t begin, uint64_t end, Func&& func) {
		if (begin >= end) {
			return;
		}
		const auto first = Start(begin);
		if ((end - first + WindowSize - 1) / WindowSize <= ScanWindows) {
			for (auto address = first; address < end; address += WindowSize) {
				if (const auto index = Find(address); index != None && Active(index)) {
					func(index);
				}
			}
			return;
		}
		for (size_t index = 0; index < Slots; index++) {
			if (!Active(index)) {
				continue;
			}
			const auto start = Begin(index);
			if (start < end && first <= start) {
				func(index);
			}
		}
	}

private:
	[[nodiscard]] bool Evictable(const Window& window, uint64_t frame) const noexcept {
		if (window.queued || window.in_flight) {
			return false;
		}
		const auto age = frame - std::min(frame, window.last_frame);
		if (window.blocked) {
			return age > BlockedFrames;
		}
		if (window.hits >= m_settings.hot_hits && !window.asleep) {
			return age > HotFrames;
		}
		return true;
	}

	size_t Insert(uint64_t address, uint64_t frame) {
		const auto home   = Home(address);
		size_t     victim = None;
		for (size_t probe = 0; probe < m_settings.probes; probe++) {
			const auto  index  = (home + probe) % Slots;
			const auto& window = m_windows[index];
			if (window.key.load(std::memory_order_relaxed) == 0) {
				victim = index;
				break;
			}
			if (Evictable(window, frame) && (victim == None || window.last_frame < m_windows[victim].last_frame)) {
				victim = index;
			}
		}
		if (victim == None) {
			return None;
		}
		auto& window = m_windows[victim];
		window.key.store(Start(address) + 1, std::memory_order_relaxed);
		window.touched.store(0, std::memory_order_relaxed);
		window.last_frame = frame;
		window.hit_frame  = frame;
		window.hits       = 0;
		window.idle       = 0;
		window.wake       = 0;
		window.last_idle  = false;
		window.asleep     = false;
		window.blocked    = false;
		window.again      = false;
		return victim;
	}

	Settings                  m_settings {};
	std::array<Window, Slots> m_windows {};
};

}

#endif
