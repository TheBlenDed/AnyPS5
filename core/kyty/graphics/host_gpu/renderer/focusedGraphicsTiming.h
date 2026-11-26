#pragma once

#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include <array>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace Libs::Graphics {

inline uint64_t FocusPixelShader() {
	static const uint64_t shader = []() -> uint64_t {
		const char* value = std::getenv("KYTY_FOCUS_PIXEL_SHADER");
		if (!value) return 0;
		const auto end = value + std::string_view(value).size();
		uint64_t hash = 0;
		const auto parsed = std::from_chars(value, end, hash, 16);
		return parsed.ec == std::errc {} && parsed.ptr == end ? hash : 0;
	}();
	return shader;
}

struct FocusDrawState {
	uint64_t module;
	bool plain;
	uint32_t topology;
	std::array<uint64_t, 3> vertex_hashes;
	bool operator==(const FocusDrawState&) const = default;
};

struct FocusGraphicsSample {
	uint64_t frame, tick;
	uint32_t token;
	uint32_t draws = 0, indirect_or_mesh_draws = 0;
	uint64_t direct_input_elements = 0;
	bool mixed_state = false;
	FocusDrawState state {};
};

inline thread_local FocusGraphicsSample* current_graphics_sample = nullptr;

class FocusGraphicsTiming {
public:
	void Open(vk::Device device, bool supported) {
		if (!FocusPixelShader() || !supported) return;
		vk::QueryPoolCreateInfo info {};
		info.queryType = vk::QueryType::ePipelineStatistics;
		info.queryCount = Capacity;
		info.pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
		    vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
		    vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
		    vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations |
		    vk::QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches |
		    vk::QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations;
		if (device.createQueryPool(&info, nullptr, &m_pool) != vk::Result::eSuccess) m_pool = nullptr;
	}
	void Reset(const CommandRecorder& recorder) const {
		if (m_pool) recorder.resetQueryPool(m_pool, Capacity);
	}
	void Begin(const CommandRecorder& recorder, uint64_t frame, uint64_t tick, uint32_t token) {
		if (m_count == Capacity) {
			++m_dropped;
			return;
		}
		auto& slot = m_samples[m_count];
		slot = {frame, tick, token};
		current_graphics_sample = &slot;
		if (m_pool) recorder.beginQuery(m_pool, m_count, {});
		++m_count;
	}
	void End(const CommandRecorder& recorder, uint32_t token) const {
		if (m_count && current_graphics_sample == &m_samples[m_count - 1] &&
		    current_graphics_sample->token == token) {
			if (m_pool) recorder.endQuery(m_pool, m_count - 1);
			current_graphics_sample = nullptr;
		}
	}
	void Save(vk::Device device, const std::filesystem::path& folder) const {
		if (!FocusPixelShader()) return;
		std::array<std::array<uint64_t, 7>, Capacity> values {};
		if (m_pool && m_count) {
			const auto result = device.getQueryPoolResults(m_pool, 0, m_count, sizeof(values), values.data(),
			    sizeof(values[0]), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			if (result != vk::Result::eSuccess && result != vk::Result::eNotReady) values = {};
		}
		std::ofstream out(folder / "focused-graphics.csv");
		out << "# dropped=" << m_dropped << ",hardware_pool=" << static_cast<bool>(m_pool) << '\n';
		out << "renderer_flip,submit_tick,sample_index,draws,indirect_or_mesh_draws,direct_input_elements,first_ps_module,first_plain,first_topology,first_pre0,first_pre1,first_pre2,mixed_draw_state,hardware_ready,ia_primitives,vs_invocations,clip_primitives,fs_invocations,tcs_patches,tes_invocations\n";
		for (uint32_t i = 0; i < m_count; ++i) {
			const auto& s = m_samples[i];
			out << s.frame << ',' << s.tick << ',' << s.token / 2 << ',' << s.draws << ',' << s.indirect_or_mesh_draws
			    << ',' << s.direct_input_elements << ',' << std::hex << s.state.module << std::dec << ','
			    << s.state.plain << ',' << s.state.topology;
			for (auto hash : s.state.vertex_hashes) out << ',' << std::hex << hash << std::dec;
			out << ',' << s.mixed_state << ',' << (values[i][6] != 0);
			for (size_t j = 0; j < 6; ++j) {
				out << ',';
				if (values[i][6]) out << values[i][j];
			}
			out << '\n';
		}
	}
	void Close(vk::Device device) {
		if (m_pool) device.destroyQueryPool(m_pool, nullptr);
		m_pool = nullptr;
		m_count = m_dropped = 0;
	}
private:
	static constexpr uint32_t Capacity = 64;
	vk::QueryPool m_pool {};
	uint32_t m_count = 0, m_dropped = 0;
	std::array<FocusGraphicsSample, Capacity> m_samples {};
};

} // namespace Libs::Graphics
#endif
