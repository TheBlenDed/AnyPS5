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
	static constexpr uint64_t HotFrames     = 64;
	static constexpr uint64_t BlockedFrames = 4096;
	static constexpr uint64_t ScanWindows   = 8;
	static constexpr size_t   None          = SIZE_MAX;

	enum class Fault : uint8_t { Cold, Hot, Woken, Blocked, Dropped };
	enum class Published : uint8_t { Touched, Idle, Asleep };

	struct Window {
		std::atomic<uint64_t> key {0};
		std::atomic<uint8_t>  touched {0};
		uint64_t              last_frame = 0;
		uint32_t              hits       = 0;
		uint32_t              idle       = 0;
		bool                  asleep     = false;
		bool                  blocked    = false;
		bool                  queued     = false;
		bool                  in_flight  = false;
		bool                  again      = false;
	};

	EagerReadbackWindows()                                       = default;
	EagerReadbackWindows(const EagerReadbackWindows&)            = delete;
	EagerReadbackWindows& operator=(const EagerReadbackWindows&) = delete;

	[[nodiscard]] static constexpr uint64_t Start(uint64_t address) noexcept { return address & ~(WindowSize - 1); }
	[[nodiscard]] static constexpr size_t   Home(uint64_t address) noexcept {
		return static_cast<size_t>(((address / WindowSize) * 0x9E3779B97F4A7C15ull) >> 56) % Slots;
	}

	[[nodiscard]] Window&       At(size_t index) noexcept { return m_windows[index]; }
	[[nodiscard]] const Window& At(size_t index) const noexcept { return m_windows[index]; }
	[[nodiscard]] uint64_t      Begin(size_t index) const noexcept {
		return m_windows[index].key.load(std::memory_order_relaxed) - 1;
	}

	[[nodiscard]] size_t Find(uint64_t address) const noexcept {
		const auto key  = Start(address) + 1;
		const auto home = Home(address);
		for (size_t probe = 0; probe < Probes; probe++) {
			const auto index = (home + probe) % Slots;
			if (m_windows[index].key.load(std::memory_order_relaxed) == key) {
				return index;
			}
		}
		return None;
	}

	[[nodiscard]] bool Active(size_t index) const noexcept {
		const auto& window = m_windows[index];
		return window.key.load(std::memory_order_relaxed) != 0 && window.hits >= HotHits && !window.asleep && !window.blocked;
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
			window.asleep     = false;
			window.idle       = 0;
			window.hits       = HotHits;
			window.last_frame = frame;
			return Fault::Woken;
		}
		if (window.hits < HotHits && window.hits != 0 && frame - window.last_frame > HitFrames) {
			window.hits = 0;
		}
		window.hits       = std::min(window.hits + 1, HotHits);
		window.idle       = 0;
		window.last_frame = frame;
		return window.hits >= HotHits ? Fault::Hot : Fault::Cold;
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
			window.last_frame  = frame;
		}
		return blocked;
	}

	void NoteGuestWrite(uint64_t address) noexcept {
		const auto key  = Start(address) + 1;
		const auto home = Home(address);
		for (size_t probe = 0; probe < Probes; probe++) {
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
			window.last_frame = frame;
			return Published::Touched;
		}
		if (window.asleep || window.blocked || ++window.idle < IdleLimit) {
			return Published::Idle;
		}
		window.asleep = true;
		window.idle   = 0;
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
		if (window.hits >= HotHits && !window.asleep) {
			return age > HotFrames;
		}
		return true;
	}

	size_t Insert(uint64_t address, uint64_t frame) {
		const auto home   = Home(address);
		size_t     victim = None;
		for (size_t probe = 0; probe < Probes; probe++) {
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
		window.hits       = 0;
		window.idle       = 0;
		window.asleep     = false;
		window.blocked    = false;
		window.again      = false;
		return victim;
	}

	std::array<Window, Slots> m_windows {};
};

}

#endif
