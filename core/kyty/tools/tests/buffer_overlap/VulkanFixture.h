#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void Require(bool condition, const char* operation) {
    if (!condition) {
        std::fprintf(stderr, "Failed: %s\n", operation);
        std::exit(1);
    }
}

struct Vulkan {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family{};

    Vulkan() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &app;
        Require(vkCreateInstance(&info, nullptr, &instance) == VK_SUCCESS, "instance");
        uint32_t count{};
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        Require(count != 0, "physical devices");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties prop{};
            vkGetPhysicalDeviceProperties(candidate, &prop);
            if (!physical || prop.vendorID == 0x10de) physical = candidate;
            if (prop.vendorID == 0x10de) break;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        std::printf("Vulkan device %s\n", properties.deviceName);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
        const auto graphics = std::ranges::find_if(families, [](const auto& q) {
            return (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        });
        Require(graphics != families.end(), "graphics queue");
        family = static_cast<uint32_t>(graphics - families.begin());
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        timeline.timelineSemaphore = VK_TRUE;
        VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        create.pNext = &timeline;
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queue_info;
        Require(vkCreateDevice(physical, &create, nullptr, &device) == VK_SUCCESS, "device");
        vkGetDeviceQueue(device, family, 0, &queue);
    }

    ~Vulkan() {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};

struct GpuBuffer {
    void* mapped{};
    VkBuffer handle{};
    VkDeviceMemory memory{};

    GpuBuffer(Vulkan& vk,uint64_t bytes) {
        VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        create.size = bytes;
        create.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        Require(vkCreateBuffer(vk.device, &create, nullptr, &handle) == VK_SUCCESS, "buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(vk.device, handle, &requirements);
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;

        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(vk.physical,&props);
        const VkMemoryPropertyFlags needed=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t i=0;
        for(;i<props.memoryTypeCount;++i) if((requirements.memoryTypeBits&(1u<<i)) && (props.memoryTypes[i].propertyFlags&needed)==needed) break;
        Require(i<props.memoryTypeCount,"host coherent memory type");
        allocate.memoryTypeIndex=i;
        Require(vkAllocateMemory(vk.device, &allocate, nullptr, &memory) == VK_SUCCESS, "buffer memory");
        Require(vkBindBufferMemory(vk.device, handle, memory, 0) == VK_SUCCESS, "bind buffer");
        Require(vkMapMemory(vk.device,memory,0,bytes,0,&mapped)==VK_SUCCESS,"map");
    }

    void Release(VkDevice device) {
        vkUnmapMemory(device,memory);
        vkDestroyBuffer(device, handle, nullptr);
        vkFreeMemory(device, memory, nullptr);
    }
};
