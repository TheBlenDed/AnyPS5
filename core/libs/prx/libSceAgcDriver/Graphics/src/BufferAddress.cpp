#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <cstdint>
#include <cstdio>

namespace AgcDriver::Graphics {

VkDeviceAddress Buffer::DeviceAddress(std::source_location location) const {
    if (deviceAddress == 0) {
        char details[1024];
        std::snprintf(details, sizeof(details), "buffer has no device address (vk=0x%llx size=%zu capacity=%zu usage=0x%x properties=0x%x; caller %s:%u %s)", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(buffer)), size, capacity, static_cast<unsigned>(usage), static_cast<unsigned>(properties), location.file_name(), static_cast<unsigned>(location.line()), location.function_name());
        Require(deviceAddress != 0, details);
    }
    return deviceAddress;
}

void Buffer::initializeAddress(VkBufferUsageFlags usage) {
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) == 0) return;
    const VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer};
    deviceAddress = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &info);
    Require(deviceAddress != 0, "vkGetBufferDeviceAddressKHR returned a null address");
}

}
