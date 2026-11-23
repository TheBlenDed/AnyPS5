#include "graphics/host_gpu/renderer/rayTracingCheck.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/rt_check_compare_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shaderBindings.h"
#include "kytyHostGpuStamp.h"
#include "kytyShaderStoreStamp.h"

#include <fmt/format.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr uint64_t RegionBytes   = uint64_t {128} << 20u;
constexpr uint64_t MinRegion     = uint64_t {4} << 20u;
constexpr uint32_t QueuedAtMost  = 4;
constexpr uint32_t DeferredAtMost = 4096;
constexpr uint64_t OutputAlign   = 256;
constexpr uint32_t ResultBytes   = 32;
constexpr uint32_t VerifiedAfter = 4;
constexpr uint32_t HostFullAfter = 8;
constexpr uint32_t SlowerAfter   = 16;
constexpr uint32_t CompareGroups = 1024;
constexpr size_t   KeptFractions = 4096;
constexpr uint64_t ContextServed = 52;
constexpr uint64_t ContextCounts = 24;

struct ComparePush {
	uint32_t count = 0;
	uint32_t slot  = 0;
};

std::string Hash(uint64_t hash) {
	return fmt::format("0x{:016x}", hash);
}

std::string Spread(std::vector<double> values) {
	if (values.empty()) {
		return "none";
	}
	std::sort(values.begin(), values.end());
	return fmt::format("min {:.3g}, median {:.3g}, max {:.3g}", values.front(), values[values.size() / 2],
	                   values.back());
}

std::string Compact(std::vector<double> values) {
	std::sort(values.begin(), values.end());
	return fmt::format("{:.3g}/{:.3g}/{:.3g}", values.front(), values[values.size() / 2], values.back());
}

auto OutputOrder(const auto& output) {
	return std::tuple {output.key, output.kind, output.size};
}

auto OutputPlace(const auto& output) {
	return std::tuple {output.key, output.kind};
}

std::string VersionLine() {
	return RayTracingCheckVersionLine(KYTY_SHADER_STORE_STAMP, KYTY_HOST_GPU_STAMP,
	                                  Config::RayTracingOracle(), Config::RayTracingCheckPpm());
}

}

RayTracingCheck::RayTracingCheck(RenderContext& context): m_context(context) {}

RayTracingCheck::~RayTracingCheck() {
	const auto device = m_context.GetGraphics().device;
	if (m_pipeline) {
		device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout) {
		device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout) {
		device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

bool RayTracingCheck::Served(const ShaderModule& module) {
	const auto& info = module.info.info;
	return !module.guest_traversal && info.bvh && info.ray_query;
}

bool RayTracingCheck::Empty(const ShaderModule& module) {
	const auto& info = module.info.info;
	return !module.guest_traversal && info.bvh && !info.ray_query && RayTracing::NativeTraversals();
}

void RayTracingCheck::NoteEmpty(uint64_t hash) {
	std::lock_guard lock(m_mutex);
	Load();
	auto& program = m_programs[hash];
	if (program.empty) {
		return;
	}
	program.empty = true;
	Note(fmt::format("[RT] check {}: empty: no form and no host query serves its BVH walks, their rays miss; not checked",
	                 Hash(hash)));
	Save();
}

bool RayTracingCheck::FallsBack(uint64_t shader_hash) {
	if (!m_any_fallback.load(std::memory_order_relaxed)) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	const auto found = m_programs.find(shader_hash);
	return found != m_programs.end() && found->second.fallback;
}

void RayTracingCheck::Load() {
	if (m_loaded) {
		return;
	}
	m_loaded = true;
	const auto title = PipelineCacheTitleId();
	if (!title.empty()) {
		m_path = std::filesystem::path("_PipelineCache") / (title + ".rt");
	}
	if (m_path.empty()) {
		return;
	}
	std::ifstream in(m_path);
	std::string   version;
	if (!in || !std::getline(in, version)) {
		return;
	}
	if (version != VersionLine()) {
		Note(fmt::format("[RT] check: ignoring decisions in {} from a different build or check configuration", m_path.string()));
		return;
	}
	uint32_t verified = 0, fallback = 0, unreplayable = 0;
	for (std::string line; std::getline(in, line);) {
		std::istringstream fields(line);
		std::string        hash_text, state;
		fields >> hash_text >> state;
		const auto hash = std::strtoull(hash_text.c_str(), nullptr, 16);
		if (hash == 0) {
			continue;
		}
		auto& program = m_programs[hash];
		for (std::string token; fields >> token;) {
			if (token.starts_with("checks=")) {
				program.checks = static_cast<uint32_t>(std::strtoul(token.c_str() + 7, nullptr, 10));
			} else if (token.starts_with("unknown=")) {
				program.unknown = static_cast<uint32_t>(std::strtoul(token.c_str() + 8, nullptr, 10));
			} else if (token.starts_with("full=")) {
				program.full = static_cast<uint32_t>(std::strtoul(token.c_str() + 5, nullptr, 10));
			} else if (token.starts_with("transient=")) {
				char* rest            = nullptr;
				program.transient     = static_cast<uint32_t>(std::strtoul(token.c_str() + 10, &rest, 10));
				program.transient_max = rest != nullptr && *rest == '/' ? std::strtod(rest + 1, nullptr) : 0.0;
			} else if (token.starts_with("reason=")) {
				std::string rest;
				std::getline(fields, rest);
				program.unreplayable = token.substr(7) + rest;
				break;
			}
		}
		if (state == "verified") {
			program.verified = true;
			program.clean    = VerifiedAfter;
			verified++;
		} else if (state == "fallback") {
			program.unreplayable.clear();
			fallback++;
		} else if (state == "unreplayable" && !program.unreplayable.empty()) {
			unreplayable++;
		} else if (state == "unjudged-host-full") {
			program.host_full = true;
			program.unreplayable.clear();
		} else if (state == "empty") {
			program.unreplayable.clear();
		} else {
			program.unreplayable.clear();
		}
	}
	if (verified + fallback + unreplayable != 0) {
		Note(fmt::format("[RT] check: {} decisions read from {} ({} verified, {} not replayable, {} fell back in an "
		                 "earlier session and are checked again)",
		                 verified + fallback + unreplayable, m_path.string(), verified, unreplayable, fallback));
	}
}

void RayTracingCheck::Save() const {
	if (m_path.empty()) {
		return;
	}
	std::vector<std::pair<uint64_t, const Program*>> rows;
	for (const auto& [hash, program]: m_programs) {
		if (program.fallback || program.verified || !program.unreplayable.empty() || program.transient != 0 ||
		    program.unknown != 0 || program.full != 0 || program.empty) {
			rows.emplace_back(hash, &program);
		}
	}
	std::sort(rows.begin(), rows.end());
	std::error_code error;
	std::filesystem::create_directories(m_path.parent_path(), error);
	auto temporary = m_path;
	temporary += ".tmp";
	{
		std::ofstream out(temporary, std::ios::trunc);
		out << VersionLine() << '\n';
		for (const auto& [hash, program]: rows) {
			const auto* state = program->fallback       ? "fallback"
			                    : program->verified     ? "verified"
			                    : !program->unreplayable.empty() ? "unreplayable"
			                    : program->host_full             ? "unjudged-host-full"
			                    : program->empty                 ? "empty"
			                                                     : "checking";
			auto        row   = fmt::format("{} {} checks={}", Hash(hash), state, program->checks);
			if (!program->fractions.empty()) {
				row += " fractions=" + Compact(program->fractions);
			}
			if (program->transient != 0) {
				row += fmt::format(" transient={}/{:.3g}", program->transient, program->transient_max);
			}
			if (program->unknown != 0) {
				row += fmt::format(" unknown={}", program->unknown);
			}
			if (program->full != 0) {
				row += fmt::format(" full={}", program->full);
			}
			if (!program->unreplayable.empty() && !program->fallback && !program->verified) {
				row += " reason=" + program->unreplayable;
			}
			out << row << '\n';
		}
	}
	std::filesystem::rename(temporary, m_path, error);
}

bool RayTracingCheck::Prepare() {
	if (m_ready) {
		return true;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	m_results = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
	                                     uint64_t {ResultBytes} * m_pending.size());

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	                     "create RT check descriptor layout");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(ComparePush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	                     "create RT check pipeline layout");
	m_pipeline = CreateComputePipeline(graphics.device, RT_CHECK_COMPARE_SPV, m_pipeline_layout,
	                                   "create RT check pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "RT Check Compare");
	m_ready = true;
	Note(fmt::format("[RT] check: differential check on, one dispatch in {} per program served natively (one in {} once "
	                "verified), threshold {:.3g} of the output dwords, fallback after {} checked dispatches in a row "
	                "above it",
	                Config::RayTracingCheckPeriod(), uint64_t {Config::RayTracingCheckPeriod()} * SlowerAfter,
	                Config::RayTracingCheckPpm() / 1e6, Config::RayTracingCheckStrikes()));
	return true;
}

void RayTracingCheck::Reserve(uint64_t region) {
	if (region <= m_capacity) {
		return;
	}
	auto capacity = MinRegion;
	while (capacity < region) {
		capacity <<= 1u;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	if (m_scratch != nullptr) {
		scheduler.DeferHostOperation([retired = std::move(m_scratch)]() mutable { retired.reset(); });
	}
	m_scratch  = std::make_shared<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 3 * capacity);
	m_capacity = capacity;
	SetVulkanObjectNameF(graphics.device, m_scratch->Handle(), "RT Check Scratch");
	Note(fmt::format("[RT] check: {} MiB of device memory for checked outputs of up to {} MiB", (3 * capacity) >> 20u,
	                 capacity >> 20u));
}

uint32_t RayTracingCheck::TakeSlot() {
	for (uint32_t slot = 0; slot < m_pending.size(); slot++) {
		if (!m_pending[slot].busy) {
			return slot;
		}
	}
	return UINT32_MAX;
}

void RayTracingCheck::Note(const std::string& line) const {
	::Log::WriteToConsoleAndLog(line + "\n");
}

void RayTracingCheck::Refuse(uint64_t hash, const std::string& reason) {
	auto& program = m_programs[hash];
	if (!program.unreplayable.empty()) {
		return;
	}
	program.unreplayable = reason;
	Note(fmt::format("[RT] check {}: not replayable ({}): no differential check, the static decision stands",
	                Hash(hash), reason));
	Save();
}

bool RayTracingCheck::Sample(const ShaderModule& module, const CommandBuffer& buffer) {
	const auto period = Config::RayTracingCheckPeriod();
	if (period == 0 || m_active) {
		return false;
	}
	if (!Served(module)) {
		if (Empty(module)) {
			NoteEmpty(module.info.shader_hash);
		}
		return false;
	}
	if (&buffer != &m_context.GetCommandScheduler().Current()) {
		return false;
	}
	const auto      hash = module.info.shader_hash;
	std::lock_guard lock(m_mutex);
	Load();
	auto&           program = m_programs[hash];
	const auto      index   = program.dispatches++;
	if (program.fallback || !program.unreplayable.empty()) {
		return false;
	}
	if (const char* reason = RayTracingCheckUnreplayable(module.info)) {
		Refuse(hash, reason);
		return false;
	}
	if (program.waiting) {
		return true;
	}
	const auto every = program.verified ? uint64_t {period} * SlowerAfter : uint64_t {period};
	return index % every == 0;
}

void RayTracingCheck::Defer(uint64_t shader_hash, bool queued) {
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[shader_hash];
	program.waiting         = true;
	program.deferred++;
	program.queued += queued ? 1 : 0;
	if (program.queued > QueuedAtMost || program.deferred > DeferredAtMost) {
		program.waiting = false;
		Refuse(shader_hash, fmt::format("the emulated variant was not ready after {} dispatches and {} background "
		                                "translations",
		                                program.deferred, program.queued));
	}
}

bool RayTracingCheck::Begin(uint64_t shader_hash) {
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[shader_hash];
	const auto      slot    = TakeSlot();
	if (slot == UINT32_MAX) {
		program.skipped++;
		program.waiting = true;
		return false;
	}
	if (!Prepare()) {
		return false;
	}
	if (program.waiting && program.deferred != 0) {
		Note(fmt::format("[RT] check {}: emulated variant ready after {} deferred dispatches", Hash(shader_hash),
		                 program.deferred));
	}
	program.waiting  = false;
	program.deferred = 0;
	program.queued   = 0;
	m_pending[slot]  = {.busy = true, .hash = shader_hash, .dwords = 0, .outputs = 0, .index = program.dispatches - 1};
	m_slot           = slot;
	m_hash           = shader_hash;
	m_active         = true;
	m_prepared       = m_context.GetRayTracing()->LastPreparation();
	return true;
}

bool RayTracingCheck::Collect(const PreparedBindings& bindings, std::vector<Output>& outputs,
                              std::string& reason) const {
	const auto& program  = bindings.runtime->module->info;
	auto&       textures = m_context.GetTextureCache();
	outputs.clear();
	for (const auto& resource: program.info.buffers) {
		if (!resource.written) {
			continue;
		}
		const auto index = resource.descriptor_index;
		if (index >= bindings.buffers.size() || index >= bindings.buffer_sources.size()) {
			reason = "writes a buffer without a binding";
			return false;
		}
		const auto& source = bindings.buffer_sources[index];
		const auto& view   = bindings.buffers[index];
		if (source.address == 0 || view.buffer == nullptr || view.range == 0) {
			continue;
		}
		if (view.range == VK_WHOLE_SIZE || view.range < source.size) {
			reason = "writes a buffer without a bounded view";
			return false;
		}
		outputs.push_back({.key = source.address, .size = source.size, .kind = 0, .buffer = view.buffer,
		                   .offset = view.offset + (view.range - source.size)});
	}
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		const auto& binding = bindings.images[i];
		if (!program.info.images[i].written || !binding.image_id) {
			continue;
		}
		auto& image = textures.GetImage(binding.image_id);
		if (image.info.IsDepth() || image.info.samples != 1) {
			reason = "writes a depth or multisampled image";
			return false;
		}
		outputs.push_back({.key = image.info.data.address, .size = image.info.LinearSize(), .kind = 1,
		                   .image = binding.image_id.index});
	}
	std::sort(outputs.begin(), outputs.end(),
	          [](const Output& a, const Output& b) { return OutputOrder(a) < OutputOrder(b); });
	outputs.erase(std::unique(outputs.begin(), outputs.end(),
	                          [](const Output& a, const Output& b) { return OutputOrder(a) == OutputOrder(b); }),
	              outputs.end());
	uint64_t at = 0;
	for (auto& output: outputs) {
		output.at = at;
		at += (output.size + OutputAlign - 1) / OutputAlign * OutputAlign;
	}
	if (at == 0) {
		reason = "writes no output";
		return false;
	}
	if (at > RegionBytes) {
		reason = fmt::format("writes {} MiB, more than the {} MiB the check keeps", (at + (1u << 20u) - 1) >> 20u,
		                     RegionBytes >> 20u);
		return false;
	}
	return true;
}

void RayTracingCheck::Barrier(const CommandRecorder& command) const {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eHost;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
	                        vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

void RayTracingCheck::Copy(const CommandRecorder& command, const std::vector<Output>& outputs, uint64_t region,
                           bool restore) {
	auto&      textures = m_context.GetTextureCache();
	const auto scratch  = m_scratch->Handle();
	for (const auto& output: outputs) {
		const auto at = region + output.at;
		if (output.kind == 0) {
			if (restore) {
				const vk::BufferCopy copy {at, output.offset, output.size};
				command.copyBuffer(scratch, output.buffer, 1, &copy);
			} else {
				const vk::BufferCopy copy {output.offset, at, output.size};
				command.copyBuffer(output.buffer, scratch, 1, &copy);
			}
			continue;
		}
		auto&      image  = textures.GetImage(ImageId {output.image});
		const auto copies = image.info.BufferCopies(at);
		if (restore) {
			image.Upload(copies, scratch, at, output.size);
		} else {
			image.Download(copies, scratch, at, output.size);
		}
	}
}

void RayTracingCheck::Before(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	std::string reason;
	if (!Collect(bindings, m_outputs, reason)) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason);
		m_pending[m_slot].busy = false;
		m_active               = false;
		return;
	}
	m_region = m_outputs.back().at + (m_outputs.back().size + OutputAlign - 1) / OutputAlign * OutputAlign;
	Reserve(m_region);
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	Barrier(command);
	command.fillBuffer(m_scratch->Handle(), m_capacity, m_region, 0);
	command.fillBuffer(m_scratch->Handle(), 2 * m_capacity, m_region, 0);
	Copy(command, m_outputs, 0, false);
	Barrier(scheduler.Current().Handle());
}

void RayTracingCheck::AfterOracle(const PreparedBindings&, bool counted) {
	if (!m_active) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	const auto context = m_context.GetRayTracing()->ContextBuffer();
	counted            = counted && context;
	if (counted) {
		const vk::BufferCopy copy {ContextServed, uint64_t {m_slot} * ResultBytes + 8, ContextCounts};
		scheduler.Current().Handle().copyBuffer(context, m_results->Handle(), 1, &copy);
	}
	{
		const auto      preparation = m_context.GetRayTracing()->LastPreparation();
		std::lock_guard lock(m_mutex);
		m_pending[m_slot].counted     = counted;
		m_pending[m_slot].preparation = counted && preparation != m_prepared ? preparation : 0;
	}
	Copy(scheduler.Current().Handle(), m_outputs, m_capacity, false);
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), m_outputs, 0, true);
	Barrier(scheduler.Current().Handle());
}

void RayTracingCheck::AfterGuest(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	m_active = false;
	std::vector<Output> guest;
	std::string         reason;
	bool                same = Collect(bindings, guest, reason);
	std::vector<Output> matched;
	for (const auto& output: m_outputs) {
		const auto found = std::find_if(guest.begin(), guest.end(), [&](const Output& candidate) {
			return OutputPlace(candidate) == OutputPlace(output) && candidate.size >= output.size;
		});
		if (!same || found == guest.end()) {
			same = false;
			break;
		}
		auto& place = matched.emplace_back(*found);
		place.size  = output.size;
		place.at    = output.at;
	}
	if (!same) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason.empty() ? std::string("the emulated variant writes other outputs") : reason);
		m_pending[m_slot].busy = false;
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Barrier(scheduler.Current().Handle());
	Copy(scheduler.Current().Handle(), matched, 2 * m_capacity, false);
	const auto command = scheduler.Current().Handle();
	const auto offset  = uint64_t {m_slot} * ResultBytes;
	command.fillBuffer(m_results->Handle(), offset, 4, 0);
	command.fillBuffer(m_results->Handle(), offset + 4, 4, 0xffffffffu);
	Barrier(command);
	const std::array<vk::DescriptorBufferInfo, 3> infos {
	    vk::DescriptorBufferInfo {m_scratch->Handle(), m_capacity, m_region},
	    vk::DescriptorBufferInfo {m_scratch->Handle(), 2 * m_capacity, m_region},
	    vk::DescriptorBufferInfo {m_results->Handle(), 0, VK_WHOLE_SIZE},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const ComparePush push {.count = static_cast<uint32_t>(m_region / 4), .slot = m_slot};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(std::min<uint32_t>(CompareGroups, (push.count + 255u) / 256u), 1, 1);
	Barrier(command);
	{
		std::lock_guard lock(m_mutex);
		m_pending[m_slot].dwords  = push.count;
		m_pending[m_slot].outputs = static_cast<uint32_t>(matched.size());
	}
	const auto slot = m_slot;
	scheduler.DeferHostOperation([weak = weak_from_this(), slot] {
		if (const auto self = weak.lock()) {
			self->Finish(slot);
		}
	});
}

void RayTracingCheck::Finish(uint32_t slot) {
	m_results->Invalidate(uint64_t {slot} * ResultBytes, ResultBytes);
	const auto* words  = reinterpret_cast<const uint32_t*>(m_results->Mapped().data() + uint64_t {slot} * ResultBytes);
	const auto  differ   = words[0];
	const auto  first    = words[1];
	const auto  served     = words[2];
	const auto  returned   = words[3];
	const auto  resumes    = words[5];
	const auto  waves      = words[6];
	const auto  candidates = words[7];
	std::lock_guard lock(m_mutex);
	const auto pending     = m_pending[slot];
	m_pending[slot].busy   = false;
	auto&        program   = m_programs[pending.hash];
	const double fraction  = pending.dwords == 0 ? 0.0 : static_cast<double>(differ) / static_cast<double>(pending.dwords);
	const double threshold = Config::RayTracingCheckPpm() / 1e6;
	const bool   over      = fraction > threshold;
	const auto   outcome   = pending.preparation == 0 ? std::nullopt
	                                                  : m_context.GetRayTracing()->Outcome(pending.preparation);
	if (pending.preparation != 0 && !outcome) {
		program.unknown++;
		program.full_run = 0;
		Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}), the host readback of this "
		                 "dispatch is gone, not judged; {} unknown so far",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, fraction, program.unknown));
		Save();
		return;
	}
	if (outcome && outcome->full) {
		program.full++;
		program.full_run++;
		Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}), the host structure was full "
		                 "(more instances than it holds), not judged; {} so far",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, fraction, program.full));
		if (!program.host_full && program.full_run >= HostFullAfter) {
			program.host_full = true;
			Note(fmt::format("[RT] check {}: unjudged-host-full: {} checks in a row on a full host structure",
			                 Hash(pending.hash), program.full_run));
		}
		Save();
		return;
	}
	program.full_run = 0;
	if (outcome && (outcome->missing != 0 || outcome->uncached)) {
		program.transient++;
		program.transient_max = std::max(program.transient_max, fraction);
		Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}), transient: the host TLAS of "
		                 "this dispatch was incomplete ({} instances without a BLAS{}), not judged; {} transient checks so "
		                 "far, worst {:.3g}",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, fraction, outcome->missing,
		                 outcome->uncached ? ", TLAS memory not cached yet" : "", program.transient,
		                 program.transient_max));
		Save();
		return;
	}
	program.checks++;
	program.strikes = over ? program.strikes + 1 : 0;
	program.clean += over ? 0 : 1;
	if (program.fractions.size() < KeptFractions) {
		program.fractions.push_back(fraction);
	}
	Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}, threshold {:.3g}){}, {} "
	                "output(s){}",
	                Hash(pending.hash), pending.index, differ, pending.dwords, fraction, threshold,
	                differ == 0 ? std::string() : fmt::format(", first at dword {}", first), pending.outputs,
	                pending.counted
	                    ? fmt::format("; host: {} rays served in {} waves, {} candidates, {} resume rounds, {} box calls "
	                                  "handed back to the guest walk",
	                                  served, waves, candidates, resumes, returned)
	                    : std::string()));
	if (!program.verified && !over && program.clean >= VerifiedAfter) {
		program.verified = true;
		Note(fmt::format("[RT] check {}: verified on {} checked dispatches ({}); one dispatch in {} checked from now on",
		                Hash(pending.hash), program.checks, Spread(program.fractions),
		                uint64_t {Config::RayTracingCheckPeriod()} * SlowerAfter));
		Save();
	}
	if (!program.fallback && program.strikes >= Config::RayTracingCheckStrikes()) {
		program.fallback = true;
		Note(fmt::format("[RT] check {}: falls back to the guest BVH walk for the rest of the session: {} checked "
		                "dispatches in a row above {:.3g} ({} over {} checks)",
		                Hash(pending.hash), program.strikes, threshold, Spread(program.fractions), program.checks));
		m_any_fallback.store(true, std::memory_order_relaxed);
		Save();
	}
}

#if defined(ANYPS5_KYTY)
namespace {

constexpr uint32_t MaxProbeRules = 32;
constexpr uint32_t MaxProbePairs = 6;
std::atomic_uint64_t witness_frame {UINT64_MAX};
std::atomic_uint64_t witness_sample_frame {UINT64_MAX};
std::atomic_uint64_t pixel_history_frame {UINT64_MAX};

constexpr uint32_t PixelHistoryScale = 10000;
std::atomic_uint64_t pixel_history_region {(5000ull << 48u) | (7000ull << 32u)};

bool PixelHistoryRepeat() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_REPEAT");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

struct PixelHistoryRequest {
	uint64_t id = 0;
	uint32_t x = 0;
	uint32_t y = 0;
	uint32_t width = 0;
	uint32_t height = 0;
};

PixelHistoryRequest ReadPixelHistoryRequest(const char* directory) {
	std::ifstream stream(std::filesystem::path(directory) / "pixel-history-request.txt");
	PixelHistoryRequest request;
	if (!(stream >> request.id >> request.x >> request.y) ||
	    request.x > PixelHistoryScale || request.y > PixelHistoryScale) {
		return {};
	}
	stream >> std::ws;
	if (!stream.eof() && (!(stream >> request.width >> request.height) ||
	                      request.width < 300 || request.width > 2667 ||
	                      request.height < 400 || request.height > 3556)) {
		return {};
	}
	return request;
}

bool ColorWitnessEnabled() {
	static const bool enabled = [] {
		const auto* env = std::getenv("KYTY_COLOR_CLEAR_WITNESS");
		return env != nullptr && std::string_view(env) == "1";
	}();
	return enabled;
}

uint32_t ColorWitnessFrames() {
	static const auto frames = [] {
		const auto* env = std::getenv("KYTY_COLOR_CLEAR_WITNESS_FRAMES");
		if (env == nullptr) {
			return 1u;
		}
		const std::string_view value(env);
		uint32_t count = 0;
		const auto parse = std::from_chars(value.data(), value.data() + value.size(), count);
		return parse.ec == std::errc {} && parse.ptr == value.data() + value.size() &&
		               count >= 1 && count <= 240
		           ? count : 1u;
	}();
	return frames;
}

bool ColorWitnessAlways() {
	static const bool enabled = [] {
		const auto* env = std::getenv("KYTY_COLOR_CLEAR_WITNESS_MODE");
		return env != nullptr && std::string_view(env) == "always";
	}();
	return enabled;
}

bool ResourceProbeArmed(std::string_view directory) {
	if (directory.empty()) {
		return false;
	}
	std::error_code error;
	const auto bytes = std::filesystem::file_size(std::filesystem::path(directory) / "markers.jsonl", error);
	return !error && bytes != 0;
}

constexpr uint32_t MaxPairCopies = 128;
constexpr size_t MaxPairCandidates = 16;
constexpr uint64_t MaxPairBytes = 2u << 20u;
constexpr uint64_t MaxProbeFrames = 120;

struct ResourceProbe {
	uint64_t hash = 0;
	uint32_t binding = 0;
	uint32_t element = 0;
	uint32_t pair = 0;
	bool after = false;
	bool done = false;
};

template <typename T>
bool ParseProbeNumber(std::string_view value, T& result, int base = 10) {
	if (value.empty()) {
		return false;
	}
	const auto end = value.data() + value.size();
	const auto parsed = std::from_chars(value.data(), end, result, base);
	return parsed.ec == std::errc {} && parsed.ptr == end;
}

struct DrawProbeMatch {
	uint32_t indices = 0;
	uint64_t pixel_hash = 0;
};

const std::vector<DrawProbeMatch>& DrawProbeMatches() {
	static const auto matches = [] {
		std::vector<DrawProbeMatch> result;
		const auto* config = std::getenv("KYTY_RESOURCE_PROBE_DRAW_MATCHES");
		if (config == nullptr || *config == '\0') return result;
		std::istringstream stream(config);
		for (std::string record; std::getline(stream, record, ';');) {
			const std::string_view value(record);
			const auto separator = value.find(':');
			DrawProbeMatch match;
			if (separator == value.npos || result.size() == MaxProbeRules ||
			    !ParseProbeNumber(value.substr(0, separator), match.indices) || match.indices == 0 ||
			    !ParseProbeNumber(value.substr(separator + 1), match.pixel_hash, 16) || match.pixel_hash == 0 ||
			    std::ranges::any_of(result, [&](const auto& seen) { return seen.indices == match.indices; })) {
				::Log::WriteToConsoleAndLog("[ResourceProbe] Expected distinct index_count:pixel_hash matches\n");
				return std::vector<DrawProbeMatch> {{}};
			}
			result.push_back(match);
		}
		return result;
	}();
	return matches;
}

std::string DrawProbePrefix(uint64_t frame, uint64_t tick, uint32_t indices) {
	return DrawProbeMatches().empty() ? fmt::format("draw-{}-{}", frame, tick)
	                                  : fmt::format("draw-{}-{}-{}", frame, tick, indices);
}

std::vector<ResourceProbe> ReadResourceProbes() {
	const auto* config = std::getenv("KYTY_RESOURCE_PROBE");
	if (config == nullptr || *config == '\0') {
		return {};
	}
	std::vector<ResourceProbe> rules;
	std::array<std::array<unsigned, 2>, MaxProbePairs + 1> pair_points {};
	std::istringstream stream(config);
	for (std::string record; std::getline(stream, record, ';');) {
		const auto first = record.find(':');
		const auto second = record.find(':', first == std::string::npos ? first : first + 1);
		const auto third = record.find(':', second == std::string::npos ? second : second + 1);
		const auto fourth = record.find(':', third == std::string::npos ? third : third + 1);
		const std::string_view value(record);
		ResourceProbe probe;
		if (third == std::string::npos || rules.size() == MaxProbeRules ||
		    !ParseProbeNumber(value.substr(0, first), probe.hash, 16) ||
		    !ParseProbeNumber(value.substr(second + 1, third - second - 1), probe.binding) ||
		    !ParseProbeNumber(value.substr(third + 1, fourth - third - 1), probe.element) ||
		    (fourth != std::string::npos &&
		     (!ParseProbeNumber(value.substr(fourth + 1), probe.pair) || probe.pair == 0 ||
		      probe.pair > MaxProbePairs))) {
			::Log::WriteToConsoleAndLog(
			    "[ResourceProbe] Expected up to 32 hash:before|after:binding:element[:pair] rules\n");
			return {};
		}
		const auto point = value.substr(first + 1, second - first - 1);
		if (point != "before" && point != "after") {
			::Log::WriteToConsoleAndLog("[ResourceProbe] Invalid copy point\n");
			return {};
		}
		probe.after = point == "after";
		if (probe.pair != 0) {
			++pair_points[probe.pair][probe.after];
		}
		rules.push_back(probe);
	}
	for (const auto& points: pair_points) {
		if (points != std::array {0u, 0u} && points != std::array {1u, 1u}) {
			::Log::WriteToConsoleAndLog("[ResourceProbe] Every pair needs one after and one before rule\n");
			return {};
		}
	}
	return rules;
}

void ProbeBarrier(CommandRecorder command, vk::PipelineStageFlags2 source_stage,
                  vk::AccessFlags2 source_access, vk::PipelineStageFlags2 destination_stage,
                  vk::AccessFlags2 destination_access) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask = source_stage;
	barrier.srcAccessMask = source_access;
	barrier.dstStageMask = destination_stage;
	barrier.dstAccessMask = destination_access;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers = &barrier;
	command.pipelineBarrier2(dependency);
}

struct ProbeBufferRange {
	vk::Buffer buffer;
	uint64_t address = 0;
	uint64_t size = 0;
	uint64_t offset = 0;
};

std::optional<ProbeBufferRange> FindProbeBuffer(const PreparedBindings& stage, const ResourceProbe& probe) {
	namespace IR = ShaderRecompiler::IR;
	const auto shader_stage = stage.runtime->module->info.stage;
	if (probe.binding != IR::NativeBinding(shader_stage, IR::DescriptorBindingKind::Buffers) ||
	    probe.element >= stage.buffers.size() || probe.element >= stage.buffer_sources.size()) {
		return std::nullopt;
	}
	const auto& view = stage.buffers[probe.element];
	const auto& guest = stage.buffer_sources[probe.element];
	if (!view.buffer || !guest.address || !guest.size || view.range == VK_WHOLE_SIZE || view.range < guest.size) {
		return std::nullopt;
	}
	const uint64_t offset = view.offset + view.range - guest.size;
	if (((offset | guest.size) & 3u) != 0) {
		return std::nullopt;
	}
	return ProbeBufferRange {view.buffer, guest.address, guest.size, offset};
}

struct PendingProbeConsumer {
	std::filesystem::path directory;
	std::string name;
	std::string detail;
	std::vector<uint8_t> data;
};

struct ProbeSnapshot {
	std::string name;
	std::string detail;
	std::mutex mutex;
	std::vector<uint8_t> data;
	std::optional<PendingProbeConsumer> consumer;
};

void SaveProbe(const std::filesystem::path& directory, const std::string& name, const std::string& detail,
               const void* data, size_t size) {
	std::ofstream file(directory / (name + ".bin"), std::ios::binary);
	file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
	file.close();
	::Log::WriteToConsoleAndLog(
	    fmt::format("[ResourceProbe] {} {}{}\n", file ? "saved" : "write failed", name, detail));
}

void CompleteProducer(ProbeSnapshot& producer, std::span<const uint8_t> data) {
	std::lock_guard lock(producer.mutex);
	if (!producer.consumer) {
		producer.data.assign(data.begin(), data.end());
		return;
	}
	const auto& consumer = *producer.consumer;
	SaveProbe(consumer.directory, producer.name, producer.detail, data.data(), data.size());
	SaveProbe(consumer.directory, consumer.name, consumer.detail, consumer.data.data(), consumer.data.size());
	producer.consumer.reset();
}

void CompleteConsumer(ProbeSnapshot& producer, const std::filesystem::path& directory, const std::string& name,
                      const std::string& detail, std::span<const uint8_t> data) {
	std::lock_guard lock(producer.mutex);
	if (producer.data.empty()) {
		producer.consumer.emplace(directory, name, detail, std::vector<uint8_t>(data.begin(), data.end()));
		return;
	}
	SaveProbe(directory, producer.name, producer.detail, producer.data.data(), producer.data.size());
	SaveProbe(directory, name, detail, data.data(), data.size());
}

bool CopyProbedResource(RenderContext& context, const PreparedBindings& stage, const ResourceProbe& probe,
                        const std::filesystem::path& directory, uint32_t draw_indices,
                        const std::shared_ptr<ProbeSnapshot>& producer = {},
                        const std::shared_ptr<ProbeSnapshot>& paired = {}, bool same_backing = false) {
	namespace IR = ShaderRecompiler::IR;
	using Stage = vk::PipelineStageFlagBits2;
	using Access = vk::AccessFlagBits2;
	const auto& module = *stage.runtime->module;
	uint64_t address = 0;
	uint64_t size = 0;
	uint64_t offset = 0;
	vk::Buffer source;
	Image* image = nullptr;
	const auto uniform_binding = IR::NativeBinding(module.info.stage, IR::DescriptorBindingKind::UniformBuffers);
	const auto flattened_binding = IR::NativeBinding(module.info.stage, IR::DescriptorBindingKind::FlattenedSrt);
	if (probe.binding == uniform_binding || probe.binding == flattened_binding) {
		const auto views = probe.binding == flattened_binding ? std::span {&stage.flattened_srt, 1u}
		                                                     : std::span {stage.uniform_buffers};
		if (probe.element >= views.size()) {
			return false;
		}
		const auto& view = views[probe.element];
		if (!view.buffer || view.range == VK_WHOLE_SIZE || ((view.offset | view.range) & 3u) != 0) {
			return false;
		}
		source = view.buffer;
		size = view.range;
		offset = view.offset;
	} else if (probe.binding == IR::NativeBinding(module.info.stage, IR::DescriptorBindingKind::Buffers)) {
		const auto range = FindProbeBuffer(stage, probe);
		if (!range) {
			return false;
		}
		source = range->buffer;
		address = range->address;
		size = range->size;
		offset = range->offset;
	} else {
		const auto& resources = module.info.info.images;
		for (size_t index = 0; index < resources.size() && index < stage.images.size(); ++index) {
			const auto& resource = resources[index];
			const auto kind = IR::DescriptorBindingForImage(resource);
			if (resource.heap || !kind || resource.mip_count != 1 ||
			    IR::NativeBinding(module.info.stage, *kind) != probe.binding ||
			    resource.descriptor_index != probe.element || !stage.images[index].image_id) {
				continue;
			}
			image = &context.GetTextureCache().GetImage(stage.images[index].image_id);
			break;
		}
		if (!image || !image->backing.image || image->info.samples != 1 ||
		    image->info.IsBlock() || image->info.IsVolume() || image->info.resources.levels != 1 ||
		    image->info.resources.layers != 1 || image->backing.state.layout == vk::ImageLayout::eUndefined) {
			return false;
		}
		address = image->info.data.address;
		size = image->info.LinearSize();
	}
	if (size == 0 || size > (32u << 20u)) {
		return false;
	}

	auto& scheduler = context.GetCommandScheduler();
	auto staging = std::make_shared<Buffer>(context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
	                                        vk::BufferUsageFlagBits::eTransferDst, size);
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	std::string shape;
	if (image) {
		const auto original = image->backing.state;
		const auto copies = image->info.BufferCopies(0);
		image->Download(copies, staging->Handle(), 0, size);
		image->Transit(original.layout, original.access_mask, {}, command);
		shape = fmt::format(" extent={}x{} format={} aspect={} row_texels={}", image->info.extent.width,
		                    image->info.extent.height, static_cast<uint32_t>(image->info.pixel_format),
		                    static_cast<uint32_t>(copies.front().imageSubresource.aspectMask),
		                    copies.front().bufferRowLength);
	} else {
		ProbeBarrier(command, Stage::eAllCommands, Access::eMemoryWrite, Stage::eCopy, Access::eTransferRead);
		const vk::BufferCopy copy {offset, 0, size};
		command.copyBuffer(source, staging->Handle(), 1, &copy);
		ProbeBarrier(command, Stage::eCopy, Access::eTransferRead, Stage::eAllCommands, Access::eMemoryWrite);
	}
	ProbeBarrier(command, Stage::eCopy, Access::eTransferWrite, Stage::eHost, Access::eHostRead);
	const auto name = fmt::format("probe-{}-{}-{:016x}-{}-{}-{}", context.FlipNumber(), scheduler.CurrentTick(),
	                              probe.hash, probe.after ? "after" : "before", probe.binding, probe.element);
	auto detail = fmt::format(" module={:016x} guest={:#x} bytes={} draw_indices={}{}", module.id, address,
	                          size, draw_indices, shape);
	if (producer) {
		producer->name = name;
		producer->detail = detail;
	}
	if (paired) {
		detail += fmt::format(" pair={} producer={} same_backing={}", probe.pair, paired->name, same_backing);
	}
	scheduler.DeferHostOperation([staging = std::move(staging), directory, size, name, detail = std::move(detail),
	                              producer, paired] {
		staging->Invalidate(0, size);
		const auto data = staging->Mapped().first(static_cast<size_t>(size));
		if (producer) {
			CompleteProducer(*producer, data);
		} else if (paired) {
			CompleteConsumer(*paired, directory, name, detail, data);
		} else {
			SaveProbe(directory, name, detail, data.data(), data.size());
		}
	});
	return true;
}

struct ProbeCandidate {
	const RenderContext* context = nullptr;
	ProbeBufferRange range;
	std::shared_ptr<ProbeSnapshot> snapshot;
};

struct ProbePairState {
	std::vector<ProbeCandidate> candidates;
	uint32_t copies = 0;
	bool done = false;
};

auto FindProbeCandidate(ProbePairState& pair, const RenderContext& context, const ProbeBufferRange& range) {
	return std::ranges::find_if(pair.candidates, [&](const ProbeCandidate& entry) {
		return entry.context == &context && entry.range.address == range.address && entry.range.size == range.size;
	});
}

void CaptureProbePair(RenderContext& context, const PreparedBindings& stage, const ResourceProbe& probe,
                      ProbePairState& pair, const std::filesystem::path& directory, uint32_t draw_indices) {
	const auto range = FindProbeBuffer(stage, probe);
	if (!range || range->size > MaxPairBytes) {
		return;
	}
	auto candidate = FindProbeCandidate(pair, context, *range);
	if (probe.after) {
		if (pair.copies == MaxPairCopies) {
			return;
		}
		auto snapshot = std::make_shared<ProbeSnapshot>();
		if (!CopyProbedResource(context, stage, probe, directory, draw_indices, snapshot)) {
			return;
		}
		if (candidate != pair.candidates.end()) {
			pair.candidates.erase(candidate);
		} else if (pair.candidates.size() == MaxPairCandidates) {
			pair.candidates.erase(pair.candidates.begin());
		}
		pair.candidates.push_back({&context, *range, std::move(snapshot)});
		++pair.copies;
		return;
	}
	if (candidate == pair.candidates.end()) {
		return;
	}
	const bool same_backing = range->buffer == candidate->range.buffer && range->offset == candidate->range.offset;
	if (CopyProbedResource(context, stage, probe, directory, draw_indices, {}, candidate->snapshot, same_backing)) {
		pair.candidates.clear();
		pair.done = true;
	}
}

struct DrawShaderProbe {
	const char* directory = nullptr;
	std::vector<uint64_t> hashes;
	std::mutex mutex;
	std::unordered_map<uint64_t, std::vector<uint32_t>> programs;
	size_t bytes = 0;
	size_t captured = 0;
	bool complete = false;
};

DrawShaderProbe& DrawShaderCapture() {
	static const auto* directory = [] {
		const auto* enabled = std::getenv("KYTY_RESOURCE_PROBE_SHADERS");
		return enabled != nullptr && std::string_view(enabled) == "1"
		           ? std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR")
		           : nullptr;
	}();
	static DrawShaderProbe capture {directory, [] {
		std::vector<uint64_t> hashes;
		const auto* value = std::getenv("KYTY_RESOURCE_PROBE_SHADER_HASHES");
		if (directory == nullptr || value == nullptr) {
			return hashes;
		}
		std::string_view fields(value);
		while (!fields.empty()) {
			const auto split = fields.find(',');
			uint64_t hash = 0;
			if (!ParseProbeNumber(fields.substr(0, split), hash, 16)) {
				::Log::WriteToConsoleAndLog("[ResourceProbe] invalid shader hash filter\n");
				return std::vector<uint64_t> {0};
			}
			hashes.push_back(hash);
			if (split == std::string_view::npos) {
				break;
			}
			fields.remove_prefix(split + 1);
		}
		return hashes;
	}()};
	return capture;
}

}

void RetainProbeShader(const ShaderModule& module, std::span<const uint32_t> spirv) {
	auto& capture = DrawShaderCapture();
	if (capture.directory == nullptr || *capture.directory == '\0' || spirv.empty() ||
	    module.info.stage == ShaderType::Compute ||
	    (!capture.hashes.empty() && std::ranges::find(capture.hashes, module.info.shader_hash) == capture.hashes.end())) {
		return;
	}
	constexpr size_t limit = 64u << 20u;
	std::lock_guard lock(capture.mutex);
	if (capture.complete || spirv.size_bytes() > limit - capture.bytes || capture.programs.contains(module.id)) {
		return;
	}
	capture.programs.emplace(module.id, std::vector<uint32_t>(spirv.begin(), spirv.end()));
	capture.bytes += spirv.size_bytes();
}

std::vector<uint32_t> PixelHistoryKeepFragments(uint64_t module_id, bool maintenance8_enabled) {
	auto& capture = DrawShaderCapture();
	std::vector<uint32_t> words;
	{
		std::lock_guard lock(capture.mutex);
		const auto found = capture.programs.find(module_id);
		if (found == capture.programs.end()) {
			return {};
		}
		words = found->second;
	}
	uint32_t replaced = 0;
	for (size_t i = 5; i < words.size();) {
		const auto size = words[i] >> 16u;
		if (size == 0 || size > words.size() - i) {
			return {};
		}
		if (size == 1 && (words[i] & 0xffffu) == spv::OpKill) {
			words[i] = (1u << 16u) | spv::OpReturn;
			++replaced;
		}
		i += size;
	}
	if (replaced == 0) {
		return {};
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	spvtools::ValidatorOptions validation;
	validation.SetUniformBufferStandardLayout(true);
	validation.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (!tools.Validate(words.data(), words.size(), validation)) {
		::Log::WriteToConsoleAndLog(fmt::format("[PixelHistoryAB] invalid alternate module {:016x}\n", module_id));
		return {};
	}
	return words;
}

void ProbeDrawShaders(RenderContext& context, const DrawRenderState& draw, const DynamicState& state,
                      uint32_t indices) {
	auto& capture = DrawShaderCapture();
	if (capture.directory == nullptr || *capture.directory == '\0') {
		return;
	}
	std::lock_guard lock(capture.mutex);
	if (capture.complete) {
		return;
	}
	const auto directory = std::filesystem::path(capture.directory);
	const auto prefix = DrawProbePrefix(context.FlipNumber(), context.GetCommandScheduler().CurrentTick(), indices);
	const auto save = [&](const ShaderModule* module, std::string_view stage) {
		if (module == nullptr) {
			return;
		}
		const auto name = fmt::format("{}-{}-{:016x}-{:016x}", prefix, stage, module->info.shader_hash, module->id);
		const auto it = capture.programs.find(module->id);
		if (it == capture.programs.end()) {
			::Log::WriteToConsoleAndLog(
			    fmt::format("[ResourceProbe] SPIR-V unavailable {} retained_bytes={}\n", name, capture.bytes));
			return;
		}
		SaveProbe(directory, name, " SPIR-V", it->second.data(), it->second.size() * sizeof(uint32_t));
	};
	const auto vertices = draw.VertexStages();
	for (size_t i = 0; i < vertices.size(); ++i) {
		save(vertices[i].stage.module, fmt::format("vertex-stage-{}", i));
	}
	if (draw.ps_active) {
		const auto* pixel = draw.ps_input_info.stage.module;
		save(draw.ps_plain && pixel != nullptr && pixel->plain != nullptr ? pixel->plain.get() : pixel, "pixel");
	}
	std::string raster = fmt::format("depth_test={} depth_write={} depth_compare={} cull={} front_face={} stencil={}\n",
	                                 state.depth_test_enabled, state.depth_write_enabled,
	                                 static_cast<uint32_t>(state.depth_compare_op),
	                                 static_cast<uint32_t>(state.cull_mode),
	                                 static_cast<uint32_t>(state.front_face), state.stencil_test_enabled);
	for (const auto& viewport: std::span(state.viewports).first(state.num_viewports)) {
		raster += fmt::format("viewport {} {} {} {} {} {}\n", viewport.x, viewport.y, viewport.width, viewport.height,
		                      viewport.minDepth, viewport.maxDepth);
	}
	for (const auto& scissor: std::span(state.scissors).first(state.num_scissors)) {
		raster += fmt::format("scissor {} {} {} {}\n", scissor.offset.x, scissor.offset.y, scissor.extent.width,
		                      scissor.extent.height);
	}
	SaveProbe(directory, prefix + "-raster", "", raster.data(), raster.size());
	if (++capture.captured >= std::max<size_t>(1, DrawProbeMatches().size())) {
		capture.complete = true;
		capture.programs.clear();
	}
}

bool IsProbeColorWitnessFrame(const RenderContext& context) {
	return ColorWitnessEnabled() && context.FlipNumber() == witness_frame.load(std::memory_order_relaxed);
}

bool ApplyProbeColorWitness(RenderContext& context, vk::ClearColorValue& value) {
	if (!ColorWitnessEnabled() || value.float32[0] != 0.0f || value.float32[1] != 0.0f) {
		return false;
	}
	const auto frame = context.FlipNumber();
	const bool always = ColorWitnessAlways();
	auto chosen = witness_frame.load(std::memory_order_relaxed);
	if (chosen == UINT64_MAX) {
		const auto* directory = std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR");
		if (directory != nullptr && ResourceProbeArmed(directory)) {
			if (witness_frame.compare_exchange_strong(chosen, frame, std::memory_order_relaxed)) {
				chosen = frame;
			}
		} else if (!always) {
			return false;
		}
	}
	const uint32_t duration = ColorWitnessFrames();
	if (!always && (frame < chosen || frame - chosen >= duration)) {
		return false;
	}
	const uint32_t sample_delay = always ? 24u : std::min(24u, duration - 1);
	if (chosen != UINT64_MAX && frame >= chosen && frame - chosen >= sample_delay) {
		uint64_t unset = UINT64_MAX;
		witness_sample_frame.compare_exchange_strong(unset, frame, std::memory_order_relaxed);
	}
	value.float32[0] = std::bit_cast<float>(0x7fcaa000u);
	value.float32[1] = std::bit_cast<float>(0xffd54000u);
	return true;
}

void SaveProbeColorWitness(RenderContext& context, Image& image) {
	static std::atomic_bool recorded = false;
	const auto* directory = std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR");
	if (directory == nullptr || witness_frame.load(std::memory_order_relaxed) == UINT64_MAX ||
	    recorded.exchange(true, std::memory_order_relaxed)) {
		return;
	}
	auto& scheduler = context.GetCommandScheduler();
	const auto size = image.info.LinearSize();
	const auto copies = image.info.BufferCopies(0);
	auto staging = std::make_shared<Buffer>(context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
	                                        vk::BufferUsageFlagBits::eTransferDst, size);
	const auto state = image.backing.state;
	image.Download(copies, staging->Handle(), 0, size);
	const auto command = scheduler.Current().Handle();
	image.Transit(state.layout, state.access_mask, {}, command);
	ProbeBarrier(command, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
	             vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
	const auto name = fmt::format("witness-{}-{}-after-clear", context.FlipNumber(), scheduler.CurrentTick());
	const auto detail = fmt::format(" guest={:#x} bytes={} extent={}x{} row_texels={}", image.info.data.address,
	                                size, image.info.extent.width, image.info.extent.height,
	                                copies.front().bufferRowLength);
	::Log::WriteToConsoleAndLog(
	    fmt::format("[ResourceProbe] NaN clear control mode={} frames={} {}{}\n",
	                ColorWitnessAlways() ? "always" : "timed", ColorWitnessAlways() ? 0 : ColorWitnessFrames(),
	                name, detail));
	scheduler.DeferHostOperation([staging = std::move(staging), directory = std::filesystem::path(directory),
	                              name, detail, size] {
		staging->Invalidate(0, size);
		SaveProbe(directory, name, detail, staging->Mapped().data(), size);
	});
}

bool ProbeShaderResources(RenderContext& context, std::span<PreparedBindings* const> stages, bool after,
                          uint32_t draw_indices) {
	static auto probes = ReadResourceProbes();
	static const uint32_t draw_filter = [] {
		const auto* env = std::getenv("KYTY_RESOURCE_PROBE_DRAW_INDICES");
		uint32_t count = 0;
		return env != nullptr && ParseProbeNumber(std::string_view(env), count) ? count : 0u;
	}();
	static const auto* directory = std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR");
	static const bool match_draw_frame = [] {
		const auto* env = std::getenv("KYTY_RESOURCE_PROBE_MATCH_DRAW_FRAME");
		return env != nullptr && std::string_view(env) == "1";
	}();
	static std::atomic_bool finished = false;
	const bool repeat_resolve = match_draw_frame && PixelHistoryRepeat();
	if (probes.empty() || directory == nullptr || *directory == '\0' ||
	    (!repeat_resolve && finished.load(std::memory_order_relaxed))) {
		return false;
	}
	static std::mutex mutex;
	static std::chrono::steady_clock::time_point next_marker_check;
	static bool armed = false;
	static uint64_t armed_frame = 0;
	static uint64_t captured_draw_frame = UINT64_MAX;
	static std::array<ProbePairState, MaxProbePairs + 1> pairs;
	static uint64_t resolve_frame = UINT64_MAX;
	if (repeat_resolve && (!after || context.FlipNumber() != pixel_history_frame.load(std::memory_order_relaxed))) {
		return false;
	}
	std::lock_guard lock(mutex);
	if (repeat_resolve && resolve_frame != context.FlipNumber()) {
		resolve_frame = context.FlipNumber();
		for (auto& probe: probes) {
			probe.done = probe.pair != 0 || !probe.after;
		}
		armed = true;
		armed_frame = resolve_frame;
	}
	const auto& draw_matches = DrawProbeMatches();
	if (!after && draw_indices != 0 && !draw_matches.empty()) {
		const auto match = std::ranges::find(draw_matches, draw_indices, &DrawProbeMatch::indices);
		if (match == draw_matches.end() ||
		    (captured_draw_frame != UINT64_MAX && captured_draw_frame != context.FlipNumber()) ||
		    std::ranges::none_of(stages, [&](const PreparedBindings* stage) {
			    const auto* module = stage != nullptr && stage->runtime != nullptr ? stage->runtime->module : nullptr;
			    return module != nullptr && module->info.stage == ShaderType::Pixel &&
			           module->info.shader_hash == match->pixel_hash;
		    })) {
			return false;
		}
	}
	if (armed && context.FlipNumber() - armed_frame >= MaxProbeFrames) {
		for (size_t index = 1; index < pairs.size(); ++index) {
			auto& pair = pairs[index];
			if (pair.copies != 0 && !pair.done) {
				::Log::WriteToConsoleAndLog(
				    fmt::format("[ResourceProbe] pair={} no matching consumer after {} producer copies\n",
				                index, pair.copies));
			}
			pair.candidates.clear();
		}
		finished.store(true, std::memory_order_relaxed);
		return false;
	}
	bool captured_vertex = false;
	for (const auto* stage: stages) {
		if (!stage || !stage->runtime || !stage->runtime->module) {
			continue;
		}
		const bool vertex = stage->runtime->module->info.stage == ShaderType::Vertex;
		if (vertex && draw_filter != 0 && draw_indices != draw_filter) {
			continue;
		}
		bool save_shader_data = false;
		for (auto& probe: probes) {
			if (probe.done || (probe.pair != 0 && pairs[probe.pair].done) ||
			    probe.after != after || stage->runtime->module->info.shader_hash != probe.hash) {
				continue;
			}
			if (!vertex && (match_draw_frame || ColorWitnessEnabled())) {
				const auto history_frame = pixel_history_frame.load(std::memory_order_relaxed);
				const auto draw_frame = captured_draw_frame == UINT64_MAX ? history_frame : captured_draw_frame;
				const uint64_t sample_frame = match_draw_frame
				                                  ? draw_frame
				                                  : witness_sample_frame.load(std::memory_order_relaxed);
				if (context.FlipNumber() != sample_frame) {
					continue;
				}
			}
			if (!armed) {
				const auto now = std::chrono::steady_clock::now();
				if (now < next_marker_check) {
					return false;
				}
				next_marker_check = now + std::chrono::milliseconds(100);
				if (!ResourceProbeArmed(directory)) {
					return false;
				}
				armed = true;
				armed_frame = context.FlipNumber();
			}
			if (probe.pair != 0) {
				CaptureProbePair(context, *stage, probe, pairs[probe.pair], directory, draw_indices);
				continue;
			}
			probe.done = true;
			const bool copied = CopyProbedResource(context, *stage, probe, directory, draw_indices);
			if (!copied) {
				::Log::WriteToConsoleAndLog(fmt::format("[ResourceProbe] unavailable {:016x} {} {}[{}]\n", probe.hash,
				                                         after ? "after" : "before", probe.binding, probe.element));
			}
			if (!after && draw_indices != 0 && vertex && (copied || !draw_matches.empty())) {
				captured_vertex = true;
				save_shader_data = true;
				captured_draw_frame = context.FlipNumber();
			}
		}
		if (save_shader_data && !stage->shader_data.empty()) {
			const auto name = fmt::format("{}-shader-data-{:016x}",
			                              DrawProbePrefix(context.FlipNumber(), context.GetCommandScheduler().CurrentTick(),
			                                              draw_indices),
			                              stage->runtime->module->info.shader_hash);
			SaveProbe(directory, name, "", stage->shader_data.data(), stage->shader_data.size() * sizeof(uint32_t));
		}
	}
	finished.store(std::ranges::all_of(probes, [&](const ResourceProbe& probe) {
		return probe.done || (probe.pair != 0 && pairs[probe.pair].done);
	}), std::memory_order_relaxed);
	return captured_vertex;
}

void ProbeDrawBuffer(RenderContext& context, const char* kind, uint32_t slot, vk::Buffer buffer,
                     uint64_t offset, uint64_t size, uint32_t stride, int32_t vertex_offset,
                     uint32_t first_instance, uint32_t indices) {
	const auto* directory = std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR");
	if (!buffer || size == 0 || size > MaxPairBytes || ((size | offset) & 3u) != 0 ||
	    directory == nullptr || *directory == '\0') {
		return;
	}
	using Stage = vk::PipelineStageFlagBits2;
	using Access = vk::AccessFlagBits2;
	auto& scheduler = context.GetCommandScheduler();
	auto staging = std::make_shared<Buffer>(context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
	                                        vk::BufferUsageFlagBits::eTransferDst, size);
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	ProbeBarrier(command, Stage::eAllCommands, Access::eMemoryWrite, Stage::eCopy, Access::eTransferRead);
	const vk::BufferCopy copy {offset, 0, size};
	command.copyBuffer(buffer, staging->Handle(), 1, &copy);
	ProbeBarrier(command, Stage::eCopy, Access::eTransferRead, Stage::eAllCommands, Access::eMemoryWrite);
	ProbeBarrier(command, Stage::eCopy, Access::eTransferWrite, Stage::eHost, Access::eHostRead);
	auto name = fmt::format("{}-{}-{}", DrawProbePrefix(context.FlipNumber(), scheduler.CurrentTick(), indices),
	                        kind, slot);
	auto detail = fmt::format(" bytes={} stride={} vertex_offset={} first_instance={}", size, stride,
	                          vertex_offset, first_instance);
	scheduler.DeferHostOperation([staging = std::move(staging), path = std::filesystem::path(directory),
	                              name = std::move(name), detail = std::move(detail), size] {
		staging->Invalidate(0, size);
		SaveProbe(path, name, detail, staging->Mapped().data(), size);
	});
}

static void ProbePixelHistoryAlphaMips(RenderContext& context, const PreparedBindings& stage,
                                       const char* directory, uint32_t indices) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_CAPTURE_ALPHA_MIPS");
		return value != nullptr && std::string_view(value) == "1";
	}();
	if (!enabled || !PixelHistoryDiscardProbeIndices(indices) || stage.images.empty()) {
		return;
	}
	const auto& binding = stage.images.front();
	if (!binding.image_id || !binding.image_view) {
		return;
	}
	auto& image = context.GetTextureCache().GetImage(binding.image_id);
	const auto& info = image.info;
	if (!image.backing.image || info.IsVolume() || info.samples != 1 || info.resources.layers != 1 ||
	    info.resources.levels < 2 || info.extent.width > 4096 || info.extent.height > 4096 ||
	    (info.pixel_format != vk::Format::eBc3UnormBlock && info.pixel_format != vk::Format::eBc3SrgbBlock)) {
		return;
	}
	const auto view = std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
	if (view == image.views.end()) {
		return;
	}
	const auto frame = context.FlipNumber();
	thread_local uint64_t previous = UINT64_MAX;
	thread_local ImageId captured {};
	if (previous == frame && captured == binding.image_id) {
		return;
	}
	previous = frame;
	captured = binding.image_id;
	auto& scheduler = context.GetCommandScheduler();
	for (uint32_t level = 0; level < 2; ++level) {
		const auto original = image.backing.subresource_states.empty() ? image.backing.state :
		                      image.backing.subresource_states[level];
		if (original.layout == vk::ImageLayout::eUndefined || original.layout == vk::ImageLayout::ePreinitialized) {
			continue;
		}
		const auto extent = info.MipExtent(level);
		const auto size = uint64_t {(extent.width + 3) / 4} * ((extent.height + 3) / 4) * 16;
		auto staging = std::make_shared<Buffer>(context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst, size);
		scheduler.EndRendering();
		const auto command = scheduler.Current().Handle();
		const ImageSubresourceRange range {level, 1, 0, 1};
		image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
		const vk::BufferImageCopy copy {0, 0, 0, {vk::ImageAspectFlagBits::eColor, level, 0, 1},
		                                {0, 0, 0}, {extent.width, extent.height, 1}};
		command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                          staging->Handle(), 1, &copy);
		image.Transit(original.layout, original.access_mask, range, command);
		ProbeBarrier(command, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
		             vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
		auto name = fmt::format("cloth-alpha-{}-{}-mip{}", frame, binding.image_id.index, level);
		auto detail = fmt::format(" origin=0x0 extent={}x{} address={:#x} format={} view_base_mip={}",
		                          extent.width, extent.height, info.data.address,
		                          static_cast<uint32_t>(info.pixel_format), view->info.base_level);
		scheduler.DeferHostOperation([staging = std::move(staging), path = std::filesystem::path(directory),
		                              name = std::move(name), detail = std::move(detail), size] {
			staging->Invalidate(0, size);
			SaveProbe(path, name, detail, staging->Mapped().data(), size);
		});
	}
}

void ProbePixelHistoryShaderInputs(RenderContext& context, std::span<PreparedBindings* const> stages,
                                   uint32_t capture, uint32_t indices, bool plain_pixel) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_SHADER_INPUTS");
		return value != nullptr && std::string_view(value) == "1";
	}();
	const auto* directory = std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR");
	if (!enabled || directory == nullptr || *directory == '\0') {
		return;
	}
	const auto frame = context.FlipNumber();
	const auto tick = context.GetCommandScheduler().CurrentTick();
	const auto prefix = DrawProbePrefix(frame, tick, indices);
	uint64_t budget = 2u << 20u;
	for (size_t stage_slot = 0; stage_slot < stages.size(); ++stage_slot) {
		const auto* stage = stages[stage_slot];
		const auto* module = stage != nullptr && stage->runtime != nullptr ? stage->runtime->module : nullptr;
		if (module == nullptr) {
			continue;
		}
		const auto name = fmt::format("inputs-{}-stage-{}", capture, stage_slot);
		const bool plain = plain_pixel && module->info.stage == ShaderType::Pixel && module->plain != nullptr;
		if (module->info.stage == ShaderType::Pixel) {
			ProbePixelHistoryAlphaMips(context, *stage, directory, indices);
		}
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[PixelHistoryInputs] flip={} draw={} stage={} hash={:016x} module={:016x} shader_data_words={}\n",
		    frame, capture, stage_slot, module->info.shader_hash, plain ? module->plain->id : module->id,
		    stage->shader_data.size()));
		if (!stage->shader_data.empty()) {
			SaveProbe(directory, fmt::format("{}-{}-shader-data", prefix, name), "",
			          stage->shader_data.data(), stage->shader_data.size() * sizeof(uint32_t));
		}
		const auto copy = [&](std::span<const vk::DescriptorBufferInfo> views, std::string_view kind) {
			const auto buffer_name = fmt::format("{}-{}", name, kind);
			for (uint32_t slot = 0; slot < views.size(); ++slot) {
				const auto& view = views[slot];
				const auto size = view.buffer && view.range != VK_WHOLE_SIZE && (view.offset & 3u) == 0
				                      ? std::min({uint64_t {view.range}, uint64_t {256u << 10u}, budget}) & ~3ull
				                      : 0ull;
				::Log::WriteToConsoleAndLog(fmt::format(
				    "[PixelHistoryInputs] flip={} draw={} stage={} kind={} slot={} range={} copied={}\n",
				    frame, capture, stage_slot, kind, slot, view.range, size));
				if (size != 0) {
					ProbeDrawBuffer(context, buffer_name.c_str(), slot, view.buffer, view.offset, size, 0, 0, 0, indices);
					budget -= size;
				}
			}
		};
		copy(stage->uniform_buffers, "uniform");
		copy(std::span {&stage->flattened_srt, 1u}, "srt");
		copy(stage->buffers, "storage");
		for (size_t slot = 0; slot < stage->images.size(); ++slot) {
			const auto& image = stage->images[slot];
			const auto* backing = image.image_id ? &context.GetTextureCache().GetImage(image.image_id) : nullptr;
			const auto& info = backing != nullptr ? backing->info : image.desc.info;
			::Log::WriteToConsoleAndLog(fmt::format(
			    "[PixelHistoryInputs] flip={} draw={} stage={} image={} id={} view={} template={} "
			    "addr={:#x} format={} extent={}x{} levels={}\n",
			    frame, capture, stage_slot, slot, image.image_id.index, static_cast<bool>(image.image_view),
			    slot < 64 && ((stage->template_images >> slot) & 1u) != 0,
			    info.data.address, static_cast<uint32_t>(info.pixel_format), info.extent.width,
			    info.extent.height, info.resources.levels));
		}
	}
}

void TracePixelHistoryTextureBindings(RenderContext& context, const PreparedBindings& stage, uint32_t indices) {
	static const uint64_t selected_shader = [] {
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_TEXTURE_TRACE");
		uint64_t hash = 0;
		return value != nullptr && ParseProbeNumber(std::string_view(value), hash, 16) ? hash : 0u;
	}();
	if (selected_shader == 0 || stage.runtime == nullptr || stage.runtime->module == nullptr ||
	    stage.runtime->module->info.shader_hash != selected_shader || !PixelHistoryDiscardProbeIndices(indices)) {
		return;
	}
	const auto& module = *stage.runtime->module;
	const auto& snapshot = *stage.runtime->resources;
	struct BindingState {
		ShaderRecompiler::IR::DescriptorValue raw;
		ImageId image;
		vk::ImageView view;
		bool operator==(const BindingState&) const = default;
	};
	thread_local std::unordered_map<uint64_t, BindingState> previous;
	thread_local uint32_t logged = 0;
	for (size_t slot = 0; slot < std::min({stage.images.size(), snapshot.images.size(), module.info.info.images.size()});
	     ++slot) {
		const auto& resource = module.info.info.images[slot];
		const auto& raw = snapshot.images[slot];
		const auto& binding = stage.images[slot];
		const BindingState current {raw, binding.image_id, binding.image_view};
		const auto key = (uint64_t {indices} << 32u) | slot;
		if (const auto found = previous.find(key); found != previous.end() && found->second == current) {
			continue;
		}
		if (logged >= 128) {
			if (logged == 128) {
				::Log::WriteToConsoleAndLog("[PixelHistoryTexture] trace limit reached\n");
				++logged;
			}
			break;
		}
		previous.insert_or_assign(key, current);
		++logged;
		ShaderTextureResource texture;
		std::ranges::copy(raw.dwords, texture.fields);
		const auto* image = binding.image_id ? &context.GetTextureCache().GetImage(binding.image_id) : nullptr;
		const auto* actual_view = static_cast<const ImageViewInfo*>(nullptr);
		if (image != nullptr && binding.image_view) {
			const auto found = std::ranges::find(image->views, binding.image_view, &CachedImageView::view);
			if (found != image->views.end()) {
				actual_view = &found->info;
			}
		}
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[PixelHistoryTexture] flip={} indices={} module={:016x} slot={} descriptor={} source={} pc={:#x} "
		    "raw_words={} raw_null={} raw_addr={:#x} raw_format={} id={} view={} template={} "
		    "live_addr={:#x} live_format={} view_format={} base_mip={} base_layer={} "
		    "raw={:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}\n",
		    context.FlipNumber(), indices, module.id, slot, resource.descriptor_index, resource.source,
		    resource.first_use_pc, raw.dword_count, texture.IsNull(), texture.Base40(),
		    static_cast<uint32_t>(texture.Format()), binding.image_id.index, static_cast<bool>(binding.image_view),
		    slot < 64 && ((stage.template_images >> slot) & 1u) != 0,
		    image != nullptr ? image->info.data.address : 0,
		    image != nullptr ? static_cast<uint32_t>(image->info.pixel_format) : 0,
		    actual_view != nullptr ? static_cast<uint32_t>(actual_view->format) : 0,
		    actual_view != nullptr ? actual_view->base_level : 0,
		    actual_view != nullptr ? actual_view->base_layer : 0,
		    raw.dwords[0], raw.dwords[1], raw.dwords[2], raw.dwords[3], raw.dwords[4], raw.dwords[5], raw.dwords[6],
		    raw.dwords[7]));
	}
}

namespace {

const char* PixelHistoryDirectory() {
	static const auto* directory = std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR");
	return directory;
}

uint64_t PixelHistoryDepthAddress() {
	static const uint64_t address = [] {
		uint64_t parsed = 0;
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_DEPTH_ADDRESS");
		return value != nullptr && ParseProbeNumber(std::string_view(value), parsed, 16) ? parsed : 0;
	}();
	return address;
}

bool PixelHistoryIncludesColor() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_INCLUDE_COLOR");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

void SaveDrawAttachments(RenderContext& context, const DrawTargets& targets, const char* directory,
                         std::string_view prefix, bool after, bool pixel_history, Image* depth_override = nullptr) {
	auto& scheduler = context.GetCommandScheduler();
	const bool depth_history = pixel_history && PixelHistoryDepthAddress() != 0;
	const auto save = [&](ImageId id, vk::ImageAspectFlagBits aspect, uint32_t level, uint32_t layer,
	                      const char* kind, Image* override_image = nullptr) {
		if (!id && override_image == nullptr) return;
		auto& image = override_image != nullptr ? *override_image : context.GetTextureCache().GetImage(id);
		const auto& info = image.info;
		const auto original = image.backing.state;
		if (!image.backing.image || info.IsBlock() || info.IsVolume() || info.samples != 1 ||
		    info.resources.levels != 1 || info.resources.layers != 1 || level != 0 || layer != 0 ||
		    original.layout == vk::ImageLayout::eUndefined) {
			return;
		}
		uint32_t bytes = info.bytes_per_block;
		if (aspect == vk::ImageAspectFlagBits::eDepth) {
			if (info.pixel_format != vk::Format::eD32Sfloat && info.pixel_format != vk::Format::eD32SfloatS8Uint) {
				return;
			}
			bytes = 4;
		}
		const auto extent = info.MipExtent(level);
		if (extent.width < 640 || extent.height < 360 || bytes == 0) {
			return;
		}
		const auto region = pixel_history_region.load(std::memory_order_relaxed);
		const auto length = [](uint32_t units, uint32_t extent, uint32_t fallback) {
			return units == 0 ? fallback :
			       std::clamp(static_cast<uint32_t>((uint64_t {extent} * units + PixelHistoryScale / 2) /
			                                        PixelHistoryScale), 1u, extent);
		};
		const uint32_t sample_width = length((region >> 16u) & 0xffffu, extent.width, depth_history ? 256 : 128);
		const uint32_t sample_height = length(region & 0xffffu, extent.height, depth_history ? 192 : 128);
		const auto start = [](uint32_t units, uint32_t extent, uint32_t length) {
			const auto middle = static_cast<uint32_t>(uint64_t {extent} * units / PixelHistoryScale);
			return std::min(std::max(middle, length / 2) - length / 2, extent - length);
		};
		const uint32_t x = pixel_history ? start(region >> 48u, extent.width, sample_width) : extent.width / 4;
		const uint32_t width = pixel_history ? sample_width : extent.width - 2 * x;
		const uint32_t y = pixel_history ? start((region >> 32u) & 0xffffu, extent.height, sample_height) : extent.height / 3;
		const uint32_t height = pixel_history ? sample_height : extent.height - y;
		const auto size = uint64_t {width} * height * bytes;
		if (size > (8u << 20u)) return;
		auto staging = std::make_shared<Buffer>(context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst, size);
		const vk::BufferImageCopy copy {
		    0, 0, 0, {aspect, level, layer, 1},
		    {static_cast<int32_t>(x), static_cast<int32_t>(y), 0}, {width, height, 1}};
		image.Download(std::span {&copy, 1u}, staging->Handle(), 0, size);
		const auto command = scheduler.Current().Handle();
		image.Transit(original.layout, original.access_mask, {}, command);
		ProbeBarrier(command, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
		             vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
		auto name = fmt::format("{}-{}-{}", prefix, kind, after ? "after" : "before");
		auto detail = fmt::format(" origin={}x{} extent={}x{} format={} bytes_per_pixel={} image={:#x}",
		                          x, y, width, height, static_cast<uint32_t>(info.pixel_format), bytes,
		                          info.data.address);
		scheduler.DeferHostOperation([staging = std::move(staging), path = std::filesystem::path(directory),
		                              name = std::move(name), detail = std::move(detail), size] {
			staging->Invalidate(0, size);
			SaveProbe(path, name, detail, staging->Mapped().data(), size);
		});
	};
	const auto colors = std::span {targets.colors, targets.color_count};
	const auto color = std::ranges::find(colors, 0u, &RenderColorInfo::target_slot);
	if (depth_override == nullptr && (!depth_history || PixelHistoryIncludesColor()) && color != colors.end()) {
		save(color->image_id, vk::ImageAspectFlagBits::eColor, color->desc.view_info.base_level,
		     color->desc.view_info.base_layer, "gbuffer0");
	}
	const auto& depth = targets.depth;
	save(depth.image_id, vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
	     depth.desc.view_info.base_layer, "depth", depth_override);
}
}

void ProbeDrawAttachments(RenderContext& context, const DrawTargets& targets, uint32_t indices, bool after) {
	static const auto* directory = [] {
		const auto* env = std::getenv("KYTY_RESOURCE_PROBE_DRAW_ATTACHMENTS");
		return env != nullptr && std::string_view(env) == "1" ? std::getenv("APS5_VIDEOOUT_TELEMETRY_DIR") : nullptr;
	}();
	if (directory == nullptr || *directory == '\0') {
		return;
	}
	const auto prefix = DrawProbePrefix(context.FlipNumber(), context.GetCommandScheduler().CurrentTick(), indices);
	SaveDrawAttachments(context, targets, directory, prefix, after, false);
}

uint32_t BeginPixelHistory(RenderContext& context, const DrawTargets& targets, const DrawRenderState& draw,
                           uint32_t indices, bool depth_clear) {
	static const uint64_t color_address = [] {
		uint64_t address = 0;
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_COLOR_ADDRESS");
		return value != nullptr && ParseProbeNumber(std::string_view(value), address, 16) ? address : 0;
	}();
	static const uint32_t min_targets = [] {
		uint32_t count = 0;
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_MIN_TARGETS");
		return value != nullptr && ParseProbeNumber(std::string_view(value), count) ? std::max(count, 1u) : 1u;
	}();
	const auto depth_address = PixelHistoryDepthAddress();
	if (color_address == 0 && depth_address == 0) {
		return 0;
	}
	const bool split_budget = depth_address != 0 && PixelHistoryIncludesColor();
	const bool repeat = split_budget && PixelHistoryRepeat();
	static const uint32_t depth_skip = [] {
		uint32_t count = 0;
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_DEPTH_SKIP");
		return value != nullptr && ParseProbeNumber(std::string_view(value), count) ? count : 0;
	}();
	const bool color_draw = targets.color_count != 0;
	const bool color_phase = split_budget && color_draw;
	const auto flip = context.FlipNumber();
	const auto history_frame = pixel_history_frame.load(std::memory_order_relaxed);
	if (!repeat && history_frame != UINT64_MAX && history_frame != flip) {
		return 0;
	}

	const auto colors = std::span {targets.colors, targets.color_count};
	const auto color = std::ranges::find(colors, 0u, &RenderColorInfo::target_slot);
	const bool color_matches = color_address != 0 && color != colors.end() && color->image_id &&
	                           color->desc.info.data.address == color_address;
	const size_t budget = color_phase ? (repeat && color_matches ? 2 : 1) : 0;
	const auto& depth = targets.depth;
	const bool depth_matches = depth_address != 0 && depth.image_id && depth.desc.info.data.address == depth_address;
	if (depth_address == 0) {
		if (!color_matches || targets.color_count < min_targets) {
			return 0;
		}
	} else if (color_phase) {
		if (!depth_matches && !color_matches) {
			return 0;
		}
	} else if (!depth_matches || (!depth.depth_write_enable && !depth_clear)) {
		return 0;
	}
	const auto* directory = PixelHistoryDirectory();
	if (directory == nullptr || *directory == '\0') {
		return 0;
	}
	static std::mutex mutex;
	static uint32_t ordinal = 0;
	static std::array<uint32_t, 3> phase_counts {};
	static uint64_t depth_candidates = 0;
	static auto next_marker_check = std::chrono::steady_clock::time_point {};
	static uint64_t armed_depth_flip = UINT64_MAX;
	static uint64_t last_request = 0;
	static PixelHistoryRequest pending;
	constexpr std::array<uint32_t, 3> repeat_limits {64, 64, 128};
	const bool late_depth = repeat && depth_skip != 0 && budget == 0;
	const uint32_t max_draws = late_depth ? 192 : repeat ? repeat_limits[budget] :
	                           split_budget ? (color_phase ? 192 : 64) : 256;
	constexpr std::array phases {"depth-or-legacy", "color", "primary-color"};
	uint32_t capture = 0;
	uint64_t depth_order = 0;
	{
		const std::lock_guard lock(mutex);
		const auto selected = pixel_history_frame.load(std::memory_order_relaxed);
		if (repeat && selected != flip) {
			if (pending.id == 0) {
				const auto now = std::chrono::steady_clock::now();
				if (now < next_marker_check) {
					return 0;
				}
				next_marker_check = now + std::chrono::milliseconds(40);
				pending = ReadPixelHistoryRequest(directory);
				if (pending.id <= last_request) {
					pending = {};
					return 0;
				}
				armed_depth_flip = flip;
			}
			if (flip <= armed_depth_flip) {
				return 0;
			}
			last_request = pending.id;
			const uint64_t region = (uint64_t {pending.x} << 48u) | (uint64_t {pending.y} << 32u) |
			                        (uint64_t {pending.width} << 16u) | pending.height;
			pixel_history_region.store(region, std::memory_order_relaxed);
			::Log::WriteToConsoleAndLog(fmt::format("[PixelHistory] request={} flip={} center={}x{} size={}x{}\n",
			                                       pending.id, flip, pending.x, pending.y, pending.width, pending.height));
			pending = {};
			phase_counts.fill(0);
			depth_candidates = 0;
			ordinal = 0;
			pixel_history_frame.store(flip, std::memory_order_relaxed);
		} else if (selected == UINT64_MAX) {
			if (armed_depth_flip == UINT64_MAX) {
				const auto now = std::chrono::steady_clock::now();
				if (now < next_marker_check) {
					return 0;
				}
				next_marker_check = now + std::chrono::milliseconds(40);
				if (!ResourceProbeArmed(directory)) {
					return 0;
				}
				if (depth_address != 0) {
					armed_depth_flip = flip;
				}
			}
			if (depth_address != 0 && flip <= armed_depth_flip) {
				return 0;
			}
			pixel_history_frame.store(flip, std::memory_order_relaxed);
		} else if (selected != flip) {
			return 0;
		}
		if (repeat && budget == 0) {
			++depth_candidates;
			if (depth_candidates <= depth_skip) {
				return 0;
			}
		}
		depth_order = depth_candidates;
		auto& recorded = phase_counts[budget];
		if (recorded >= max_draws) {
			if (recorded == max_draws) {
				::Log::WriteToConsoleAndLog(fmt::format("[PixelHistory] flip={} capture_limit={} phase={}\n",
				                                       flip, max_draws, phases[budget]));
				++recorded;
			}
			return 0;
		}
		++recorded;
		capture = ++ordinal;
	}
	const auto vertices = draw.VertexStages();
	const auto* vertex = vertices.empty() ? nullptr : vertices.back().stage.module;
	const auto* pixel = draw.ps_active ? draw.ps_input_info.stage.module : nullptr;
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[PixelHistory] flip={} draw={} indices={} vs={:016x} ps={:016x} depth_clear={} color_targets={} color0={:#x} "
	    "depth={:#x} depth_test={} depth_write={} depth_compare={} depth_candidates={}\n",
	    flip, capture, indices, vertex != nullptr ? vertex->info.shader_hash : 0,
	    pixel != nullptr ? pixel->info.shader_hash : 0, depth_clear, targets.color_count,
	    color != colors.end() ? color->desc.info.data.address : 0, depth.desc.info.data.address,
	    depth.depth_test_enable, depth.depth_write_enable, static_cast<uint32_t>(depth.depth_compare_op), depth_order));
	SaveDrawAttachments(context, targets, directory, fmt::format("pixel-history-{}-{}", flip, capture), false, true);
	return capture;
}

void EndPixelHistory(RenderContext& context, const DrawTargets& targets, uint32_t capture) {
	if (capture == 0) {
		return;
	}
	const auto flip = pixel_history_frame.load(std::memory_order_relaxed);
	SaveDrawAttachments(context, targets, PixelHistoryDirectory(), fmt::format("pixel-history-{}-{}", flip, capture),
	                    true, true);
}

bool PixelHistoryDiscardProbeIndices(uint32_t indices) {
	static const auto selected = [] {
		const auto* value = std::getenv("KYTY_PIXEL_HISTORY_DISCARD_AB_INDICES");
		std::vector<uint32_t> counts;
		if (value == nullptr) {
			return counts;
		}
		std::string_view fields(value);
		while (!fields.empty() && counts.size() < 16) {
			const auto separator = fields.find(',');
			uint32_t count = 0;
			if (!ParseProbeNumber(fields.substr(0, separator), count) || count == 0) {
				return std::vector<uint32_t> {};
			}
			counts.push_back(count);
			if (separator == std::string_view::npos) {
				break;
			}
			fields.remove_prefix(separator + 1);
		}
		return counts;
	}();
	return std::ranges::find(selected, indices) != selected.end();
}

std::shared_ptr<Image> CopyPixelHistoryDepth(RenderContext& context, const DrawTargets& targets) {
	if (!targets.depth.image_id) {
		return {};
	}
	auto& source = context.GetTextureCache().GetImage(targets.depth.image_id);
	const auto& view = targets.depth.desc.view_info;
	const auto original = source.backing.state;
	if (!source.backing.image || source.info.samples != 1 || source.backing.mip_levels != 1 ||
	    source.backing.layers != 1 || view.base_level != 0 || view.base_layer != 0 ||
	    original.layout == vk::ImageLayout::eUndefined || source.backing.extent.width > 4096 ||
	    source.backing.extent.height > 4096 ||
	    (source.backing.format != vk::Format::eD32Sfloat && source.backing.format != vk::Format::eD32SfloatS8Uint)) {
		return {};
	}
	auto info = source.info;
	info.data = {};
	info.stencil = {};
	info.metadata = {};
	auto& scheduler = context.GetCommandScheduler();
	auto copy = std::make_shared<Image>(context.GetGraphics(), scheduler, info);
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	copy->Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	std::array<vk::ImageCopy, 2> regions {};
	for (size_t i = 0; i < regions.size(); ++i) {
		const auto aspect = i == 0 ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eStencil;
		auto& region = regions[i];
		region.srcSubresource = {aspect, 0, 0, 1};
		region.dstSubresource = region.srcSubresource;
		region.extent = source.backing.extent;
	}
	const auto count = source.backing.format == vk::Format::eD32SfloatS8Uint ? 2u : 1u;
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
	                  copy->backing.image, vk::ImageLayout::eTransferDstOptimal, count, regions.data());
	source.Transit(original.layout, original.access_mask, {}, command);
	copy->Transit(vk::ImageLayout::eDepthStencilAttachmentOptimal,
	              vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	              {}, command);
	return copy;
}

void SavePixelHistoryDiscardProbe(RenderContext& context, const DrawTargets& targets, uint32_t capture,
                                  const std::shared_ptr<Image>& copy) {
	const auto flip = context.FlipNumber();
	auto prefix = fmt::format("pixel-history-{}-{}-keep-fragments", flip, capture);
	SaveDrawAttachments(context, targets, PixelHistoryDirectory(), prefix, true, true, copy.get());
}

#endif

}
