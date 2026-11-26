#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace Libs::Graphics {

namespace {

#if defined(ANYPS5_KYTY)
bool TraceVmaBudget() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_VMA_BUDGET_TRACE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

constexpr VkDeviceSize SmallImageBytes = 64 * 1024;

struct ImageBlockReuseWindow {
	std::chrono::steady_clock::time_point refresh {};
	std::chrono::steady_clock::time_point report {};
	VmaAllocator allocator = nullptr;
	uint32_t memory_types = 0;
	uint64_t attempts = 0;
	uint64_t reused = 0;
	uint64_t spilled = 0;
	uint64_t misses = 0;
	uint64_t required = 0;
};

bool ReuseSmallImageBlocks() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_VMA_SMALL_IMAGE_REUSE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

uint32_t AvailableImageBlockTypes(VmaAllocator allocator, const VkPhysicalDeviceMemoryProperties& props) {
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	bool local_pressure = false;
	for (uint32_t i = 0; i < props.memoryHeapCount; ++i) {
		local_pressure |= (props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 &&
		                  budgets[i].budget != 0 && budgets[i].usage >= budgets[i].budget;
	}
	if (!local_pressure) {
		return 0;
	}
	uint32_t types = 0;
	constexpr auto excluded = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT |
	                          VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
		const auto& type = props.memoryTypes[i];
		const auto& stats = budgets[type.heapIndex].statistics;
		if ((type.propertyFlags & excluded) == 0 && stats.blockBytes > stats.allocationBytes &&
		    stats.blockBytes - stats.allocationBytes >= SmallImageBytes) {
			types |= 1u << i;
		}
	}
	return types;
}

bool EligibleImageBlockReuse(const VkImageCreateInfo& info) {
	constexpr auto excluded = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
	                          VK_IMAGE_CREATE_SPARSE_ALIASED_BIT | VK_IMAGE_CREATE_PROTECTED_BIT |
	                          VK_IMAGE_CREATE_DISJOINT_BIT;
	return info.pNext == nullptr && info.imageType == VK_IMAGE_TYPE_2D &&
	       info.tiling == VK_IMAGE_TILING_OPTIMAL && info.sharingMode == VK_SHARING_MODE_EXCLUSIVE &&
	       info.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED && (info.flags & excluded) == 0 &&
	       info.samples == VK_SAMPLE_COUNT_1_BIT && info.extent.width <= 256 && info.extent.height <= 256;
}

VkResult BindImageInExistingBlocks(VmaAllocator allocator, VkImage image,
                                   const VkMemoryRequirements& requirements, uint32_t types,
                                   VmaAllocation* allocation) {
	VmaAllocationCreateInfo suballocate {};
	suballocate.flags = VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
	suballocate.memoryTypeBits = types & requirements.memoryTypeBits;
	suballocate.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	if (suballocate.memoryTypeBits == 0) {
		return VK_ERROR_FEATURE_NOT_PRESENT;
	}
	const auto result = vmaAllocateMemory(allocator, &requirements, &suballocate, allocation, nullptr);
	if (result != VK_SUCCESS) {
		return result;
	}
	const auto bound = vmaBindImageMemory(allocator, *allocation, image);
	if (bound != VK_SUCCESS) {
		vmaFreeMemory(allocator, *allocation);
		*allocation = nullptr;
	}
	return bound;
}

bool TryImageBlockReuse(const GraphicContext& graphics, const VkImageCreateInfo& info,
                        VkImage* image, VmaAllocation* allocation) {
	if (!ReuseSmallImageBlocks() || !graphics.memory_budget_ext_enabled ||
	    graphics.physical_device_properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu ||
	    !EligibleImageBlockReuse(info)) {
		return false;
	}
	const auto& props = static_cast<const VkPhysicalDeviceMemoryProperties&>(graphics.physical_device_memory_properties);
	static thread_local ImageBlockReuseWindow window;
	const auto now = std::chrono::steady_clock::now();
	if (window.allocator != graphics.allocator || now >= window.refresh) {
		if (window.allocator != graphics.allocator) {
			window = {};
		}
		window.allocator = graphics.allocator;
		window.refresh = now + std::chrono::milliseconds(20);
		window.memory_types = AvailableImageBlockTypes(graphics.allocator, props);
		if (TraceVmaBudget() && window.attempts != 0 && now >= window.report) {
			window.report = now + std::chrono::milliseconds(250);
			Log::WriteToConsoleAndLog(fmt::format(
			    "[kyty-vma-image-reuse] t_ns={} attempts={} reused={} nonlocal={} misses={} required={}\n",
			    std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(),
			    window.attempts, window.reused, window.spilled, window.misses, window.required));
		}
	}
	if (window.memory_types == 0) {
		return false;
	}
	++window.attempts;
	VkImage candidate = VK_NULL_HANDLE;
	const auto device = static_cast<VkDevice>(graphics.device);
	const auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (dispatch.vkCreateImage(device, &info, nullptr, &candidate) != VK_SUCCESS) {
		return false;
	}
	VkMemoryDedicatedRequirements dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
	VkMemoryRequirements2 requirements {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedicated};
	const VkImageMemoryRequirementsInfo2 request {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, candidate};
	dispatch.vkGetImageMemoryRequirements2(device, &request, &requirements);
	uint32_t types = window.memory_types;
	constexpr auto writes = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
	                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
	if ((info.usage & writes) != 0) {
		for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
			if ((props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
				types &= ~(1u << i);
			}
		}
	}
	if (dedicated.requiresDedicatedAllocation) {
		++window.required;
	} else if (requirements.memoryRequirements.size <= SmallImageBytes) {
		const auto result = BindImageInExistingBlocks(graphics.allocator, candidate,
		                                              requirements.memoryRequirements, types, allocation);
		if (result == VK_SUCCESS) {
			*image = candidate;
			VmaAllocationInfo bound {};
			vmaGetAllocationInfo(graphics.allocator, *allocation, &bound);
			++window.reused;
			window.spilled += (props.memoryTypes[bound.memoryType].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0;
			return true;
		}
		if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
			++window.misses;
			window.memory_types = 0;
		}
	}
	dispatch.vkDestroyImage(device, candidate, nullptr);
	return false;
}

void TraceVmaBudgetSample(const GraphicContext& graphics, const VmaBudget* budgets) {
	if (!TraceVmaBudget()) {
		return;
	}
	static std::atomic<int64_t> next_sample {0};
	const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count();
	auto next = next_sample.load(std::memory_order_relaxed);
	if (now < next || !next_sample.compare_exchange_strong(next, now + 1'000'000'000,
	                                                       std::memory_order_relaxed)) {
		return;
	}
	const auto& properties = graphics.physical_device_memory_properties;
	for (uint32_t i = 0; i < properties.memoryHeapCount; ++i) {
		const auto& heap = budgets[i];
		const auto& stats = heap.statistics;
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-vma] t_ns={} heap={} local={} usage={} budget={} allocation_bytes={} "
		    "block_bytes={} allocation_count={} block_count={}\n",
		    now, i, static_cast<bool>(properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal),
		    heap.usage, heap.budget, stats.allocationBytes, stats.blockBytes, stats.allocationCount,
		    stats.blockCount));
	}
}
constexpr uint64_t SlowVmaCreateNs = 20'000'000;

struct BufferCreateTrace {
	const VkBufferCreateInfo&     request;
	const VmaAllocationCreateInfo& allocation;
	const VkBuffer*               buffer;
};

bool TraceSlowVmaBuffers() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_VMA_BUFFER_DETAIL");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

void ReportSlowVmaBuffer(VmaAllocator allocator, const BufferCreateTrace& trace,
                         const VmaAllocationInfo2& info, int64_t begin_ns, int64_t end_ns) {
	VmaAllocatorInfo owner {};
	vmaGetAllocatorInfo(allocator, &owner);
	VkMemoryDedicatedRequirements dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
	VkMemoryRequirements2 requirements {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedicated};
	const auto query = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2;
	if (query != nullptr) {
		const VkBufferMemoryRequirementsInfo2 request {VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
		                                              nullptr, *trace.buffer};
		query(owner.device, &request, &requirements);
	}
	const VkPhysicalDeviceMemoryProperties* memory = nullptr;
	vmaGetMemoryProperties(allocator, &memory);
	const auto type = info.allocationInfo.memoryType;
	const auto& memory_type = memory->memoryTypes[type];
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto& budget = budgets[memory_type.heapIndex];
	static thread_local const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
	const auto detail_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count() - end_ns;
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-vma-buffer-detail] tid={} begin_ns={} end_ns={} bytes={} usage=0x{:x} flags=0x{:x} "
	    "alloc_flags=0x{:x} alloc_usage={} required_flags=0x{:x} preferred_flags=0x{:x} "
	    "allocated={} block_bytes={} type={} heap={} memory_flags=0x{:x} usage_bytes={} budget_bytes={} "
	    "dedicated={} requirements_known={} requires_dedicated={} prefers_dedicated={} detail_ns={}\n",
	    tid, begin_ns, end_ns, trace.request.size, trace.request.usage, trace.request.flags,
	    trace.allocation.flags, static_cast<int>(trace.allocation.usage), trace.allocation.requiredFlags,
	    trace.allocation.preferredFlags, info.allocationInfo.size, info.blockSize, type,
	    memory_type.heapIndex, memory_type.propertyFlags, budget.usage, budget.budget,
	    info.dedicatedMemory != 0, query != nullptr, dedicated.requiresDedicatedAllocation != 0,
	    dedicated.prefersDedicatedAllocation != 0, detail_ns));
}

template <bool Image, typename Create>
VkResult TraceVmaCreate(VmaAllocator allocator, VmaAllocation* allocation, Create&& create,
                       const BufferCreateTrace* buffer_trace = nullptr) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_VMA_CREATE_TRACE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	if (!enabled) {
		return create();
	}

	using Clock = std::chrono::steady_clock;
	using Nanoseconds = std::chrono::nanoseconds;
	struct Interval {
		Clock::time_point begin {};
		uint64_t count = 0;
		uint64_t failures = 0;
		uint64_t dedicated = 0;
		uint64_t local_dedicated = 0;
		uint64_t local_dedicated_bytes = 0;
		uint64_t local_dedicated_ns = 0;
		uint64_t total_ns = 0;
		uint64_t max_ns = 0;
	};
	const auto begin = Clock::now();
	const auto result = create();
	const auto end = Clock::now();
	const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<Nanoseconds>(end - begin).count());
	static thread_local Interval interval {};
	if (interval.count == 0) {
		interval.begin = begin;
	}
	++interval.count;
	interval.total_ns += ns;
	interval.max_ns = std::max(interval.max_ns, ns);
	if (result == VK_SUCCESS) {
		VmaAllocationInfo2 info {};
		vmaGetAllocationInfo2(allocator, *allocation, &info);
		if (buffer_trace != nullptr && ns >= SlowVmaCreateNs) {
			ReportSlowVmaBuffer(allocator, *buffer_trace, info,
			                    std::chrono::duration_cast<Nanoseconds>(begin.time_since_epoch()).count(),
			                    std::chrono::duration_cast<Nanoseconds>(end.time_since_epoch()).count());
		}
		if (info.dedicatedMemory) {
			++interval.dedicated;
			VkMemoryPropertyFlags properties {};
			vmaGetMemoryTypeProperties(allocator, info.allocationInfo.memoryType, &properties);
			if ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
				++interval.local_dedicated;
				interval.local_dedicated_bytes += info.allocationInfo.size;
				interval.local_dedicated_ns += ns;
			}
		}
	} else {
		++interval.failures;
	}
	if (end - interval.begin >= std::chrono::milliseconds(250) || ns >= SlowVmaCreateNs) {
		const auto start_ns = std::chrono::duration_cast<Nanoseconds>(interval.begin.time_since_epoch()).count();
		const auto end_ns = std::chrono::duration_cast<Nanoseconds>(end.time_since_epoch()).count();
		static thread_local const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-vma-create] kind={} tid={} begin_ns={} end_ns={} count={} failures={} dedicated={} "
		    "local_dedicated={} local_dedicated_bytes={} local_dedicated_ns={} total_ns={} max_ns={}\n",
		    Image ? "image" : "buffer", tid, start_ns, end_ns, interval.count, interval.failures,
		    interval.dedicated, interval.local_dedicated, interval.local_dedicated_bytes,
		    interval.local_dedicated_ns, interval.total_ns, interval.max_ns));
		interval = {};
	}
	return result;
}
#endif

struct RecycledImage {
	vk::ImageCreateInfo create;
	VkImage             image      = VK_NULL_HANDLE;
	VmaAllocation       allocation = nullptr;
	uint64_t            size       = 0;
};

constexpr size_t   RecycledImagesMax     = 32;
constexpr uint64_t RecycledImageBytesMax = 256ull * 1024 * 1024;

std::mutex                 g_recycled_mutex;
std::vector<RecycledImage> g_recycled;
uint64_t                   g_recycled_bytes = 0;

bool Recyclable(const vk::ImageCreateInfo& create) {
	return create.pNext == nullptr && create.tiling == vk::ImageTiling::eOptimal &&
	       create.sharingMode == vk::SharingMode::eExclusive &&
	       create.initialLayout == vk::ImageLayout::eUndefined;
}

bool SameCreateInfo(const vk::ImageCreateInfo& a, const vk::ImageCreateInfo& b) {
	return a.flags == b.flags && a.imageType == b.imageType && a.format == b.format &&
	       a.extent == b.extent && a.mipLevels == b.mipLevels && a.arrayLayers == b.arrayLayers &&
	       a.samples == b.samples && a.usage == b.usage;
}

template <typename Handle, auto Destroy>
class ResourceGraveyard {
public:
	ResourceGraveyard() = default;
	~ResourceGraveyard() {
		Flush();
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_one();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	ResourceGraveyard(const ResourceGraveyard&)            = delete;
	ResourceGraveyard& operator=(const ResourceGraveyard&) = delete;

	bool Push(VmaAllocator allocator, Handle handle, VmaAllocation allocation, uint64_t bytes = 0) {
		constexpr uint64_t ByteLimit = 128ull * 1024 * 1024;
		constexpr size_t ItemLimit = 64;
		{
			std::lock_guard lock(m_mutex);
			if (bytes != 0 && (m_count >= ItemLimit || bytes > ByteLimit - m_bytes)) {
				return false;
			}
			if (!m_thread.joinable()) {
				m_thread = std::thread([this] { Run(); });
			}
			m_items.push_back({allocator, handle, allocation, bytes});
			++m_count;
			m_bytes += bytes;
		}
		m_wake.notify_one();
		return true;
	}

	void Flush() {
		std::unique_lock lock(m_mutex);
		m_idle.wait(lock, [this] { return m_items.empty() && !m_busy; });
	}

private:
	struct Item {
		VmaAllocator  allocator;
		Handle        handle;
		VmaAllocation allocation;
		uint64_t      bytes;
	};

	void Run() {
		std::vector<Item> batch;
		std::unique_lock  lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_items.empty()) {
				return;
			}
			batch.swap(m_items);
			m_busy = true;
			lock.unlock();
			uint64_t bytes = 0;
			for (const auto& item: batch) {
				Destroy(item.allocator, item.handle, item.allocation);
				bytes += item.bytes;
			}
			lock.lock();
			m_count -= batch.size();
			m_bytes -= bytes;
			batch.clear();
			m_busy = false;
			m_idle.notify_all();
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_wake;
	std::condition_variable m_idle;
	std::vector<Item>       m_items;
	size_t                  m_count = 0;
	uint64_t                m_bytes = 0;
	bool                    m_busy = false;
	bool                    m_stop = false;
	std::thread             m_thread;
};

auto& Graveyard() {
	static ResourceGraveyard<VkImage, vmaDestroyImage> graveyard;
	return graveyard;
}

#if defined(ANYPS5_KYTY)
auto& BufferGraveyard() {
	static ResourceGraveyard<VkBuffer, vmaDestroyBuffer> graveyard;
	return graveyard;
}
#endif

void DestroyRecycledImages(VmaAllocator allocator) {
	Graveyard().Flush();
	std::lock_guard lock(g_recycled_mutex);
	for (const auto& entry: g_recycled) {
		vmaDestroyImage(allocator, entry.image, entry.allocation);
	}
	g_recycled.clear();
	g_recycled_bytes = 0;
}

}

#if defined(ANYPS5_KYTY)
VkResult CreateVmaTracedBuffer(VmaAllocator allocator, const VkBufferCreateInfo* buffer_info,
                               const VmaAllocationCreateInfo* alloc_info, VkBuffer* buffer,
                               VmaAllocation* allocation, VmaAllocationInfo* result) {
	const BufferCreateTrace detail {*buffer_info, *alloc_info, buffer};
	return TraceVmaCreate<false>(allocator, allocation, [&] {
		return vmaCreateBuffer(allocator, buffer_info, alloc_info, buffer, allocation, result);
	}, TraceSlowVmaBuffers() ? &detail : nullptr);
}

bool TryRetireDedicatedBuffer(GraphicContext& graphics, VkBuffer buffer, VmaAllocation allocation) {
	VmaAllocationInfo2 info {};
	vmaGetAllocationInfo2(graphics.allocator, allocation, &info);
	return info.dedicatedMemory &&
	       BufferGraveyard().Push(graphics.allocator, buffer, allocation, info.blockSize);
}
#endif

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
#if defined(ANYPS5_KYTY)
	if (const auto* value = std::getenv("KYTY_VMA_BLOCK_MIB"); value && std::string_view(value) == "64") {
		info.preferredLargeHeapBlockSize = 64ull * 1024 * 1024;
	}
	if (TraceVmaBudget()) {
		Log::WriteToConsoleAndLog(fmt::format("[kyty-vma] preferred_large_block_bytes={} driver_budget={}\n",
		                                    info.preferredLargeHeapBlockSize, memory_budget_ext_enabled));
	}
#endif

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages(allocator);
#if defined(ANYPS5_KYTY)
	BufferGraveyard().Flush();
#endif
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
#if defined(ANYPS5_KYTY)
	TraceVmaBudgetSample(*this, budgets);
#endif
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	if (Recyclable(image_info)) {
		std::lock_guard lock(g_recycled_mutex);
		const auto      found =
		    std::find_if(g_recycled.begin(), g_recycled.end(), [&](const auto& entry) {
			    return SameCreateInfo(entry.create, image_info);
		    });
		if (found != g_recycled.end()) {
			native_image     = found->image;
			image.allocation = found->allocation;
			g_recycled_bytes -= found->size;
			g_recycled.erase(found);
		}
	}
	if (native_image == VK_NULL_HANDLE) {
		const auto allocate = [&] {
#if defined(ANYPS5_KYTY)
			if (TryImageBlockReuse(*this, static_cast<const VkImageCreateInfo&>(image_info),
			                       &native_image, &image.allocation)) {
				return VK_SUCCESS;
			}
#endif
			return vmaCreateImage(allocator, static_cast<const VkImageCreateInfo*>(image_info),
			                      &alloc_info, &native_image, &image.allocation, nullptr);
		};
		const auto create = [&] {
#if defined(ANYPS5_KYTY)
			return TraceVmaCreate<true>(allocator, &image.allocation, allocate);
#else
			return allocate();
#endif
		};
		auto result = create();
		if (result != VK_SUCCESS) {
			DestroyRecycledImages(allocator);
			result = create();
		}
		if (result != VK_SUCCESS) {
			image.image = native_image;
			LogMemoryBudget();
			return false;
		}
	}
	image.image = native_image;

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.recyclable = Recyclable(image_info);
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	vk::ImageCreateInfo create {};
	create.flags         = image.flags;
	create.imageType     = image.image_type;
	create.format        = image.format;
	create.extent        = image.extent;
	create.mipLevels     = image.mip_levels;
	create.arrayLayers   = image.layers;
	create.samples       = static_cast<vk::SampleCountFlagBits>(image.samples);
	create.tiling        = vk::ImageTiling::eOptimal;
	create.usage         = image.usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.initialLayout = vk::ImageLayout::eUndefined;
	if (image.recyclable) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		std::lock_guard lock(g_recycled_mutex);
		g_recycled.push_back({create, image.image, image.allocation, allocation_info.size});
		g_recycled_bytes += allocation_info.size;
		while (g_recycled.size() > RecycledImagesMax || g_recycled_bytes > RecycledImageBytesMax) {
			const auto& oldest = g_recycled.front();
			Graveyard().Push(allocator, oldest.image, oldest.allocation);
			g_recycled_bytes -= oldest.size;
			g_recycled.erase(g_recycled.begin());
		}
	} else {
		Graveyard().Push(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
}

}
