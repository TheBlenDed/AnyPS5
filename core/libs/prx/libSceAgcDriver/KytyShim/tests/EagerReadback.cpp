#include "graphics/host_gpu/renderer/cache/eagerReadbackWindows.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::EagerReadbackWindows;
using Fault     = EagerReadbackWindows::Fault;
using Published = EagerReadbackWindows::Published;

constexpr uint64_t Window = EagerReadbackWindows::WindowSize;
constexpr uint64_t Pool   = 0x240000000;
constexpr size_t   None   = EagerReadbackWindows::None;
constexpr uint64_t LooseHitFrames = 16;
constexpr uint32_t LooseIdleLimit = 8;

void Require(bool value, const char* what) {
	if (value) return;
	std::fprintf(stderr, "kyty_shim_eager_readback: FAILED: %s\n", what);
	std::exit(1);
}

std::vector<uint64_t> SameHome(size_t count) {
	std::vector<uint64_t> windows {Pool};
	const auto            home = EagerReadbackWindows::Home(Pool);
	for (auto address = Pool + Window; windows.size() < count; address += Window) {
		if (EagerReadbackWindows::Home(address) == home) windows.push_back(address);
	}
	return windows;
}

std::unique_ptr<EagerReadbackWindows> Loose() {
	EagerReadbackWindows::Settings settings;
	settings.hot_hits   = 2;
	settings.hit_frames = LooseHitFrames;
	settings.idle_limit = LooseIdleLimit;
	settings.wake_hits  = 1;
	return std::make_unique<EagerReadbackWindows>(settings);
}

void MakeHot(EagerReadbackWindows& windows, uint64_t address, uint64_t frame, size_t& index) {
	(void)windows.NoteFault(address, frame, index);
	Require(windows.NoteFault(address, frame, index) == Fault::Hot, "two faults in a frame make a window hot");
}

void Learning() {
	auto   windows = Loose();
	size_t index   = None;
	Require(windows->Find(Pool) == None, "a fresh table holds no window");
	Require(windows->ActiveCount() == 0, "a fresh table has no hot window");
	Require(windows->NoteFault(Pool + 0x1234, 10, index) == Fault::Cold, "one fault leaves a window cold");
	Require(index != None && windows->Find(Pool + Window - 1) == index, "every address of a window finds it");
	Require(windows->Begin(index) == Pool, "a window starts at its 64 KiB boundary");
	Require(windows->Find(Pool + Window) == None, "the next window is a different one");
	Require(!windows->Active(index), "a cold window is not active");
	Require(windows->NoteFault(Pool + 0x8000, 10 + LooseHitFrames, index) == Fault::Hot,
	        "a second fault within the hit frames makes the window hot");
	Require(windows->Active(index) && windows->ActiveCount() == 1, "a hot window is active");
	Require(windows->NoteFault(Pool, 5000, index) == Fault::Hot, "a hot window stays hot across a long gap");

	size_t other = None;
	Require(windows->NoteFault(Pool + Window, 10, other) == Fault::Cold, "the neighbour starts cold");
	Require(windows->NoteFault(Pool + Window, 11 + LooseHitFrames, other) == Fault::Cold,
	        "a second fault after the hit frames starts the count again");
	Require(windows->NoteFault(Pool + Window, 12 + LooseHitFrames, other) == Fault::Hot,
	        "the fault after that one makes it hot");
}

void Aging() {
	auto   windows = Loose();
	size_t index   = None;
	MakeHot(*windows, Pool, 1, index);
	Require(windows->NotePublished(index, 2) == Published::Touched, "the publication after a fault is not idle");
	for (uint32_t i = 1; i < LooseIdleLimit; i++) {
		Require(windows->NotePublished(index, 2 + i) == Published::Idle, "a publication without an access is idle");
		Require(windows->Active(index), "the window stays active below the idle limit");
	}
	windows->NoteGuestWrite(Pool + 0x4000);
	Require(windows->NotePublished(index, 20) == Published::Touched, "a guest write ends the idle streak");
	windows->NoteGuestWrite(Pool + Window);
	for (uint32_t i = 1; i < LooseIdleLimit; i++) {
		Require(windows->NotePublished(index, 20 + i) == Published::Idle, "a write to another window does not count");
	}
	Require(windows->NotePublished(index, 40) == Published::Asleep, "the idle limit puts the window to sleep");
	Require(!windows->Active(index) && windows->ActiveCount() == 0, "a sleeping window is not active");
	windows->NoteGuestWrite(Pool);
	Require(!windows->Active(index), "a guest write does not wake a sleeping window");
	Require(windows->NoteFault(Pool, 41, index) == Fault::Woken, "a fault wakes a sleeping window");
	Require(windows->Active(index), "a woken window is active");
	Require(windows->NotePublished(index, 42) == Published::Touched, "the waking fault counts as an access");
}

void Blocking() {
	auto   windows = Loose();
	size_t index   = None;
	MakeHot(*windows, Pool + Window, 1, index);
	Require(windows->Block(Pool + 0x100, Pool + 2 * Window + 1, 2) == 3, "a readback over three windows blocks three");
	Require(!windows->Active(index), "a blocked window is not active");
	Require(windows->Block(Pool, Pool + 3 * Window, 3) == 0, "blocking again blocks nothing new");
	Require(windows->NoteFault(Pool + Window, 4, index) == Fault::Blocked, "a fault leaves a blocked window blocked");
	Require(windows->NoteFault(Pool + Window, 4, index) == Fault::Blocked, "so does the fault after it");
	Require(!windows->Active(index) && windows->ActiveCount() == 0, "no blocked window turns hot");
	size_t free_window = None;
	Require(windows->NoteFault(Pool + 3 * Window, 4, free_window) == Fault::Cold, "the window after the readback is free");
}

void Eviction() {
	auto       windows = Loose();
	const auto same    = SameHome(EagerReadbackWindows::Probes + 2);
	size_t     index   = None;
	for (size_t i = 0; i < EagerReadbackWindows::Probes; i++) {
		MakeHot(*windows, same[i], 100 + i, index);
	}
	Require(windows->NoteFault(same[EagerReadbackWindows::Probes], 110, index) == Fault::Dropped,
	        "a full probe run of recent hot windows takes no new window");
	Require(windows->NoteFault(same[EagerReadbackWindows::Probes], 101 + EagerReadbackWindows::HotFrames, index) ==
	            Fault::Cold,
	        "a hot window without an access for the hot frames gives its slot up");
	Require(windows->Find(same[0]) == None, "the oldest hot window went");
	Require(windows->Find(same[1]) != None, "the newer ones stayed");

	const auto queued       = windows->Find(same[1]);
	windows->At(queued).queued = true;
	const auto flying          = windows->Find(same[2]);
	windows->At(flying).in_flight = true;
	const uint64_t late = 1000;
	Require(windows->NoteFault(same[EagerReadbackWindows::Probes + 1], late, index) == Fault::Cold,
	        "a new window takes the slot of the window without an access for longest");
	Require(windows->Find(same[3]) == None, "which was the old hot one");
	Require(windows->Find(same[1]) == queued && windows->Find(same[2]) == flying,
	        "a queued window and one with a copy in flight keep their slots");
	Require(windows->NoteFault(same[0], late, index) == Fault::Cold && windows->Find(same[EagerReadbackWindows::Probes]) == None,
	        "the next one replaces the cold window");
	windows->At(windows->Find(same[0])).queued                                 = true;
	windows->At(windows->Find(same[EagerReadbackWindows::Probes + 1])).in_flight = true;
	Require(windows->NoteFault(same[3], late + 5000, index) == Fault::Dropped && index == None,
	        "a probe run of queued windows and windows with a copy in flight takes no new window");

	auto blocked = Loose();
	for (size_t i = 0; i < EagerReadbackWindows::Probes; i++) {
		Require(blocked->Block(same[i], same[i] + 1, 100) == 1, "each window of the run is blocked");
	}
	Require(blocked->NoteFault(same[EagerReadbackWindows::Probes], 100 + EagerReadbackWindows::BlockedFrames, index) ==
	            Fault::Dropped,
	        "blocked windows keep their slots for the blocked frames");
	Require(blocked->NoteFault(same[EagerReadbackWindows::Probes], 101 + EagerReadbackWindows::BlockedFrames, index) ==
	            Fault::Cold,
	        "and give one up after them");
}

void Ranges() {
	auto                  windows = Loose();
	size_t                index   = None;
	std::vector<uint64_t> seen;
	const auto            collect = [&](uint64_t begin, uint64_t end) {
		seen.clear();
		windows->ForEachActiveIn(begin, end, [&](size_t at) { seen.push_back(windows->Begin(at)); });
	};
	MakeHot(*windows, Pool, 1, index);
	MakeHot(*windows, Pool + 5 * Window, 1, index);
	MakeHot(*windows, Pool + 40 * Window, 1, index);
	(void)windows->NoteFault(Pool + 2 * Window, 1, index);
	collect(Pool + Window - 1, Pool + Window);
	Require(seen.size() == 1 && seen[0] == Pool, "a write to the last byte of a window finds it");
	collect(Pool + Window, Pool + 5 * Window);
	Require(seen.empty(), "a write between hot windows finds none, the cold one included");
	collect(Pool + 0x10, Pool + 5 * Window + 1);
	Require(seen.size() == 2, "a short write finds the hot windows it touches");
	collect(Pool - 3 * Window, Pool + 64 * Window);
	Require(seen.size() == 3, "a long write finds every hot window it covers");
	collect(Pool + Window, Pool + 40 * Window);
	Require(seen.size() == 1 && seen[0] == Pool + 5 * Window, "a long write stops before the window after its end");
	collect(Pool, Pool);
	Require(seen.empty(), "an empty write finds nothing");
}

void GuestThreads() {
	auto   windows = Loose();
	size_t index   = None;
	MakeHot(*windows, Pool, 1, index);
	Require(windows->NotePublished(index, 2) == Published::Touched, "the first publication takes the fault");
	std::vector<std::thread> threads;
	for (int i = 0; i < 4; i++) {
		threads.emplace_back([&windows, i] {
			for (uint64_t at = 0; at < 4096; at++) {
				windows->NoteGuestWrite(Pool + (at * 16 + static_cast<uint64_t>(i)) % Window);
				windows->NoteGuestWrite(Pool + Window + at);
			}
		});
	}
	for (auto& thread: threads) thread.join();
	Require(windows->NotePublished(index, 3) == Published::Touched, "writes of other threads reach the window");
	Require(windows->NotePublished(index, 4) == Published::Idle, "and are taken once");
}

void Thresholds() {
	const auto defaults = std::make_unique<EagerReadbackWindows>();
	Require(defaults->Parameters().hot_hits == EagerReadbackWindows::HotHits &&
	            defaults->Parameters().hit_frames == EagerReadbackWindows::HitFrames &&
	            defaults->Parameters().idle_limit == EagerReadbackWindows::IdleLimit &&
	            defaults->Parameters().wake_hits == EagerReadbackWindows::WakeHits &&
	            defaults->Parameters().probes == EagerReadbackWindows::Probes,
	        "a table without settings uses the defaults");
	Require(EagerReadbackWindows::HotHits == 3 && EagerReadbackWindows::HitFrames == 2 &&
	            EagerReadbackWindows::IdleLimit == 1 && EagerReadbackWindows::WakeHits == 2,
	        "the defaults are three faults within two frames, one idle publication and two waking faults");
	EagerReadbackWindows::Settings zero;
	zero.hot_hits   = 0;
	zero.hit_frames = 0;
	zero.idle_limit = 0;
	zero.wake_hits  = 0;
	zero.probes     = 0;
	auto   lowest = std::make_unique<EagerReadbackWindows>(zero);
	size_t index  = None;
	Require(lowest->Parameters().hot_hits == 1 && lowest->Parameters().idle_limit == 1 &&
	            lowest->Parameters().wake_hits == 1 && lowest->Parameters().probes == 1,
	        "settings below one count as one");
	Require(lowest->NoteFault(Pool, 1, index) == Fault::Hot, "with one hit the first fault makes a window hot");
	zero.probes = 100000;
	Require(std::make_unique<EagerReadbackWindows>(zero)->Parameters().probes == EagerReadbackWindows::Slots,
	        "no more probes than slots");

	EagerReadbackWindows::Settings settings;
	settings.hot_hits   = 3;
	settings.hit_frames = 3;
	settings.idle_limit = 1;
	settings.wake_hits  = 2;
	settings.probes     = 8;
	auto windows = std::make_unique<EagerReadbackWindows>(settings);
	Require(windows->NoteFault(Pool, 10, index) == Fault::Cold, "the first of three faults leaves the window cold");
	Require(windows->NoteFault(Pool, 12, index) == Fault::Cold && !windows->Active(index), "so does the second");
	Require(windows->NoteFault(Pool, 13, index) == Fault::Hot && windows->Active(index),
	        "the third within the frame span makes it hot");
	size_t other = None;
	Require(windows->NoteFault(Pool + Window, 10, other) == Fault::Cold, "the neighbour takes its first fault");
	Require(windows->NoteFault(Pool + Window, 12, other) == Fault::Cold, "and its second");
	Require(windows->NoteFault(Pool + Window, 14, other) == Fault::Cold,
	        "a third fault after the frame span of the first starts the count again");
	Require(windows->NoteFault(Pool + Window, 16, other) == Fault::Cold, "the span is counted from that fault");
	Require(windows->NoteFault(Pool + Window, 17, other) == Fault::Hot, "three faults within it make the window hot");

	Require(windows->NotePublished(index, 14) == Published::Touched && !windows->At(index).last_idle,
	        "the publication after the faults is not idle");
	Require(windows->NotePublished(index, 15) == Published::Asleep && windows->At(index).last_idle,
	        "with an idle limit of one the first idle publication puts the window to sleep");
	Require(windows->NoteFault(Pool, 100, index) == Fault::Asleep && !windows->Active(index),
	        "one fault does not wake a window that needs two");
	Require(windows->NoteFault(Pool, 104, index) == Fault::Asleep && !windows->Active(index),
	        "nor does a second one after the frame span");
	Require(windows->NoteFault(Pool, 105, index) == Fault::Woken && windows->Active(index),
	        "two faults within the frame span wake it");
	Require(windows->At(index).last_idle, "a woken window still has an idle last publication");
	Require(windows->NotePublished(index, 106) == Published::Touched && !windows->At(index).last_idle,
	        "the waking faults count as an access");
	Require(windows->NotePublished(index, 107) == Published::Asleep, "the window sleeps again");
	Require(windows->NoteFault(Pool, 108, index) == Fault::Asleep, "and counts its waking faults from zero");
	Require(windows->Block(Pool, Pool + 1, 109) == 1 && windows->NoteFault(Pool, 110, index) == Fault::Blocked,
	        "a blocked sleeping window is blocked");

	const auto same = SameHome(10);
	auto       wide = std::make_unique<EagerReadbackWindows>(settings);
	for (size_t i = 0; i < 8; i++) {
		(void)wide->NoteFault(same[i], 100, index);
		(void)wide->NoteFault(same[i], 100, index);
		Require(wide->NoteFault(same[i], 100, index) == Fault::Hot, "eight probes hold eight hot windows of one home");
	}
	Require(wide->NoteFault(same[8], 101, index) == Fault::Dropped, "the ninth finds no slot");
	Require(!wide->Full(101), "while the table is not full");
	wide->NoteGuestWrite(same[7]);
	Require(wide->NotePublished(wide->Find(same[7]), 102) == Published::Touched,
	        "a guest write finds a window at the last probe");

	settings.hot_hits = 1;
	settings.probes   = EagerReadbackWindows::Slots;
	auto   full  = std::make_unique<EagerReadbackWindows>(settings);
	size_t count = 0;
	for (; count < EagerReadbackWindows::Slots; count++) {
		Require(!full->Full(200), "a table with a free slot is not full");
		Require(full->NoteFault(Pool + count * Window, 200, index) == Fault::Hot, "every slot takes a window");
	}
	Require(full->ActiveCount() == EagerReadbackWindows::Slots && full->Full(200), "a table of recent hot windows is full");
	Require(full->NoteFault(Pool + count * Window, 200, index) == Fault::Dropped, "and takes no new window");
	Require(!full->Full(201 + EagerReadbackWindows::HotFrames), "until a hot window had no access for the hot frames");
	Require(full->NoteFault(Pool + count * Window, 201 + EagerReadbackWindows::HotFrames, index) == Fault::Hot,
	        "then the new window gets a slot");
}

}

int main() {
	Learning();
	Aging();
	Blocking();
	Eviction();
	Ranges();
	GuestThreads();
	Thresholds();
	std::puts("kyty_shim_eager_readback: ok");
	return 0;
}
