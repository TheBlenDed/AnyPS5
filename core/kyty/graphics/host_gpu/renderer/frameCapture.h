#ifndef KYTY_GRAPHICS_FRAME_CAPTURE_H_
#define KYTY_GRAPHICS_FRAME_CAPTURE_H_

#if defined(ANYPS5_KYTY) && defined(_WIN32)
#include <windows.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <mutex>

namespace Libs::Graphics::FrameCapture {

using Event = std::array<uint64_t, 16>;
constexpr uint32_t Slots = 32;
constexpr uint32_t MaxEvents = 4096;

struct Slot {
	volatile LONG token;
	uint32_t count, dropped, width, height, format;
	uint64_t source, flip, tick;
	Event events[MaxEvents];
};
static_assert(sizeof(Slot) == 48 + MaxEvents * sizeof(Event));

class History {
public:
	History() {
		const auto* enabled = std::getenv("KYTY_FRAME_CAPTURE");
		if (enabled == nullptr || std::strcmp(enabled, "1") != 0) return;
		wchar_t name[64];
		std::swprintf(name, 64, L"Local\\AnyKytyFrameV1_%lu", GetCurrentProcessId());
		m_file = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, Slots * sizeof(Slot), name);
		if (m_file != nullptr && GetLastError() != ERROR_ALREADY_EXISTS) {
			m_slots = static_cast<Slot*>(MapViewOfFile(m_file, FILE_MAP_WRITE, 0, 0, 0));
		}
	}
	~History() {
		if (m_slots != nullptr) UnmapViewOfFile(m_slots);
		if (m_file != nullptr) CloseHandle(m_file);
	}
	bool Enabled() const { return m_slots != nullptr; }
	void Record(const Event& event) {
		std::lock_guard lock(m_mutex);
		if (m_count == MaxEvents) ++m_dropped;
		else m_events[m_count++] = event;
	}
	uint32_t Seal(uint64_t source, uint32_t width, uint32_t height, uint32_t format, uint64_t flip, uint64_t tick) {
		if (!Enabled()) return 0;
		std::lock_guard lock(m_mutex);
		if (m_serial == UINT32_MAX) return 0;
		const auto token = ++m_serial;
		auto& slot = m_slots[token % Slots];
		InterlockedExchange(&slot.token, 0);
		slot.count = m_count;
		slot.dropped = m_dropped;
		slot.width = width;
		slot.height = height;
		slot.format = format;
		slot.source = source;
		slot.flip = flip;
		slot.tick = tick;
		std::memcpy(slot.events, m_events.data(), m_count * sizeof(Event));
		InterlockedExchange(&slot.token, static_cast<LONG>(token));
		m_count = m_dropped = 0;
		return token;
	}
private:
	HANDLE m_file = nullptr;
	Slot* m_slots = nullptr;
	std::mutex m_mutex;
	uint32_t m_count = 0, m_dropped = 0, m_serial = 0;
	std::array<Event, MaxEvents> m_events {};
};

inline History& Get() {
	static History history;
	return history;
}

inline uint64_t Tag(uint32_t token, uint32_t pid) {
	uint8_t crc = 0xa7;
	for (uint32_t word : {pid, token}) {
		for (int i = 0; i < 32; ++i) {
			const bool bit = ((crc >> 7) ^ (word >> (31 - i))) & 1;
			crc = static_cast<uint8_t>((crc << 1) ^ (bit ? 0x07 : 0));
		}
	}
	return (uint64_t{0xd6} << 40) | (uint64_t{token} << 8) | crc;
}

} // namespace Libs::Graphics::FrameCapture
#endif
#endif
