#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler {

bool ProveHullQuadHelpersUnobserved(const IR::Program& program, const IR::Inst& mask,
                                    std::span<IR::Inst* const> reads, IR::Value predicate);

}
