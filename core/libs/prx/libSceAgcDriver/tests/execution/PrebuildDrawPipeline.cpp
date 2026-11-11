#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <future>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}

int main() {
    using namespace AgcDriver::Graphics;
    using ShaderRecompiler::ShaderStage;
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        context.graphicsPipelineLibrary = false;
        static_cast<void>(GetBufferPool(context));
        Require(context.descriptorCache != nullptr && context.pipelineCacheSynchronization != nullptr, "prebuild fixture needs production descriptor cache and synchronization");
        const auto before = PipelinePrebuildTotals();
        const auto storedBefore = PipelineStoreTotals();
        std::shared_future<void> first, duplicate;
        auto gate = context.pipelineCacheSynchronization->LockForSnapshot();
        struct Completion {
            PipelineCacheSynchronization::SnapshotLock& gate;
            std::shared_future<void>& first;
            std::shared_future<void>& duplicate;
            ~Completion() {
                if (gate.owns_lock()) gate.unlock();
                if (first.valid()) first.wait();
                if (duplicate.valid()) duplicate.wait();
            }
        } completion{gate, first, duplicate};
        {
            const std::array<std::uint32_t, 13> vertexCode{
                0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
                0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u
            };
            const std::array<std::uint32_t, 5> pixelCode{0x7e0002f2u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u};
            auto target = device->Target();
            target.fixedPushSlots = false;
            const std::vector<std::uint32_t> userData(4);
            ShaderRecompiler::ShaderVertexStageInfo vertexInfo{};
            const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(vertexCode.data()), std::as_bytes(std::span(vertexCode))}}};
            ShaderRecompiler::RecompileRequest vertexRequest{
                {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(vertexCode.data()), vertexCode, 0, {}},
                {32, 0, userData, std::nullopt, std::nullopt, vertexInfo, vertexMemory}, target, {0, 0, 0, 64}
            };
            vertexRequest.useCache = false;
            auto vertex = ShaderRecompiler::Recompile(vertexRequest);
            const auto vertexPush = static_cast<std::uint32_t>(vertex.pushConstants.size());
            ShaderRecompiler::ShaderPixelStageInfo pixelInfo{};
            pixelInfo.wave32 = true;
            pixelInfo.targetOutputMode[0] = 9;
            pixelInfo.targetExportMapping.fill(0xe4u);
            const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(pixelCode.data()), std::as_bytes(std::span(pixelCode))}}};
            ShaderRecompiler::RecompileRequest pixelRequest{
                {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(pixelCode.data()), pixelCode, 0, {}},
                {32, 0, userData, std::nullopt, pixelInfo, std::nullopt, pixelMemory}, target, PixelPushLayout(vertexPush, target)
            };
            pixelRequest.useCache = false;
            auto pixel = ShaderRecompiler::Recompile(pixelRequest);
            const std::array<CompiledShader, 2> shaders{{{ShaderStage::Vertex, &vertex, 0}, {ShaderStage::Fragment, &pixel, PixelPushOffset(vertexPush, target)}}};
            State state{};
            ColorTarget color{};
            color.format = VK_FORMAT_R8G8B8A8_UNORM;
            color.extent = {1, 1};
            state.color = color;
            state.colors = {color};
            state.hasColorTarget = true;
            state.blend.colorWriteMask = 0xf;
            state.blends = {state.blend};
            first = PrebuildDrawPipeline(context, state, shaders);
            duplicate = PrebuildDrawPipeline(context, state, shaders);
            Require(first.valid() && duplicate.valid(), "cold prebuild was not queued");
            Require(first.wait_for(std::chrono::seconds(0)) != std::future_status::ready && duplicate.wait_for(std::chrono::seconds(0)) != std::future_status::ready, "prebuild escaped the pipeline cache gate");
            const auto queued = PipelinePrebuildTotals();
            Require(queued.queued == before.queued + 1 && queued.deduped == before.deduped + 1, "identical requests did not share one queued build");
        }
        gate.unlock();
        Require(first.wait_for(std::chrono::seconds(30)) == std::future_status::ready, "prebuild did not complete after releasing the cache gate");
        first.get();
        duplicate.get();
        Require(PipelineStoreTotals().cached == storedBefore.cached + 1, "background prebuild failed to publish its pipeline");
        ClearCachedPipelines(context.device);
        std::puts("real Vulkan prebuild dedupe and caller-lifetime tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
