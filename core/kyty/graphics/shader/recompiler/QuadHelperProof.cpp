#include "graphics/shader/recompiler/QuadHelperProof.h"

#include "common/logging/log.h"
#include "Optimization/QuadHelperProof.hpp"
#include "RdnaDecoder/RdnaInstruction.hpp"

#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

namespace Native = ::ShaderRecompiler;
using IR::Inst;
using IR::Value;
using IR::ValueOpcode;
using Native::IrOpcode;
using Native::IrType;

IrType ProofType(IR::Type type) {
	switch (type) {
		case IR::Type::Void: return IrType::Void;
		case IR::Type::U1: return IrType::Bool;
		case IR::Type::U8: return IrType::U8;
		case IR::Type::U16: return IrType::U16;
		case IR::Type::U32: return IrType::U32;
		case IR::Type::U64: return IrType::U64;
		case IR::Type::F16: return IrType::F16;
		case IR::Type::F32: return IrType::F32;
		case IR::Type::U32x2: return IrType::U32x2;
		case IR::Type::U32x3: return IrType::U32x3;
		case IR::Type::U32x4: return IrType::U32x4;
		case IR::Type::ScalarReg: return IrType::ScalarReg;
		case IR::Type::VectorReg: return IrType::VectorReg;
		case IR::Type::BufferResource: return IrType::BufferResource;
		case IR::Type::ImageResource: return IrType::ImageResource;
		case IR::Type::SamplerResource: return IrType::SamplerResource;
		case IR::Type::ImageAddress: return IrType::ImageAddress;
		default: return IrType::Opaque;
	}
}

IrOpcode ProofOpcode(ValueOpcode opcode) {
	using enum IR::ValueOpcode;
	switch (opcode) {
		case Identity: case ConditionRef: return IrOpcode::Identity;
		case Phi: return IrOpcode::Phi;
		case Reference: return IrOpcode::Reference;
		case ReferenceU32: return IrOpcode::ReferenceU32;
		case LogicalAnd: return IrOpcode::LogicalAnd;
		case LogicalOr: return IrOpcode::LogicalOr;
		case LogicalXor: return IrOpcode::LogicalXor;
		case LogicalNot: return IrOpcode::LogicalNot;
		case IEqual32: return IrOpcode::IEqual32;
		case INotEqual32: return IrOpcode::INotEqual32;
		case ULessThan32: return IrOpcode::ULessThan32;
		case UGreaterThan32: return IrOpcode::UGreaterThan32;
		case UGreaterThanEqual32: return IrOpcode::UGreaterThanEqual32;
		case FPUnordGreaterThanEqual32: return IrOpcode::FPUnordGreaterThanEqual32;
		case FPOrdGreaterThanEqual32: case FPOrdLessThanEqual32: return IrOpcode::FPOrdLessThanEqual32;
		case FPOrdGreaterThan32: case FPOrdLessThan32: return IrOpcode::FPOrdLessThan32;
		case SelectU1: return IrOpcode::SelectU1;
		case SelectU32: return IrOpcode::SelectU32;
		case SelectF32: return IrOpcode::SelectF32;
		case BitCastF32U32: return IrOpcode::BitCastF32U32;
		case BitCastU32F32: return IrOpcode::BitCastU32F32;
		case CompositeConstructU32x2: return IrOpcode::CompositeConstructU32x2;
		case CompositeConstructU32x3: return IrOpcode::CompositeConstructU32x3;
		case CompositeConstructU32x4: return IrOpcode::CompositeConstructU32x4;
		case CompositeExtractU32x2: return IrOpcode::CompositeExtractU32x2;
		case CompositeExtractU32x3: return IrOpcode::CompositeExtractU32x3;
		case CompositeExtractU32x4: return IrOpcode::CompositeExtractU32x4;
		case CompositeConstructU64: return IrOpcode::CompositeConstructU64;
		case CompositeExtractU64: return IrOpcode::CompositeExtractU64;
		case Ballot: return IrOpcode::Ballot;
		case LaneId: return IrOpcode::LaneId;
		case BitwiseAnd32: return IrOpcode::BitwiseAnd32;
		case BitwiseAnd64: return IrOpcode::BitwiseAnd64;
		case BitwiseOr32: return IrOpcode::BitwiseOr32;
		case BitwiseXor32: return IrOpcode::BitwiseXor32;
		case BitwiseNot32: return IrOpcode::BitwiseNot32;
		case IAdd32: return IrOpcode::IAdd32;
		case ISub32: return IrOpcode::ISub32;
		case IMul32: return IrOpcode::IMul32;
		case UMin32: return IrOpcode::UMin32;
		case BitCount32: return IrOpcode::BitCount32;
		case BitFieldInsert: return IrOpcode::BitFieldInsert;
		case BitFieldUExtract: return IrOpcode::BitFieldUExtract;
		case ShiftLeftLogical32: return IrOpcode::ShiftLeftLogical32;
		case ShiftRightLogical32: return IrOpcode::ShiftRightLogical32;
		case ShiftRightLogical64: return IrOpcode::ShiftRightLogical64;
		case FPMad32: return IrOpcode::FPMad32;
		case FPMul32: return IrOpcode::FPMul32;
		case FPAdd32: return IrOpcode::FPAdd32;
		case FPSub32: return IrOpcode::FPSub32;
		case FPAbs32: return IrOpcode::FPAbs32;
		case FPLog2: return IrOpcode::FPLog2;
		case FPExp2: return IrOpcode::FPExp2;
		case FPCeil32: return IrOpcode::FPCeil32;
		case FPMax32: return IrOpcode::FPMax32;
		case FPMaxTri32: return IrOpcode::FPMaxTri32;
		case FPMin32: return IrOpcode::FPMin32;
		case FPMinTri32: return IrOpcode::FPMinTri32;
		case FPSaturate32: return IrOpcode::FPSaturate32;
		case FPSqrt: return IrOpcode::FPSqrt;
		case GetBufferResource: return IrOpcode::GetBufferResource;
		case GetAddressResource: return IrOpcode::GetAddressResource;
		case GetImageResource: return IrOpcode::GetImageResource;
		case GetSamplerResource: return IrOpcode::GetSamplerResource;
		case MakeImageAddress: return IrOpcode::MakeImageAddress;
		case ReadConstBuffer: return IrOpcode::ReadConstBuffer;
		case LoadAddressU32: return IrOpcode::LoadAddressU32;
		case LoadBufferU32: return IrOpcode::LoadBufferU32;
		case LoadBufferU32x2: return IrOpcode::LoadBufferU32x2;
		case LoadBufferU32x3: return IrOpcode::LoadBufferU32x3;
		case LoadBufferU32x4: return IrOpcode::LoadBufferU32x4;
		case LoadSharedU32: return IrOpcode::LoadSharedU32;
		case LoadSharedU32x3: return IrOpcode::LoadSharedU32x3;
		case ImageSampleRaw: return IrOpcode::ImageSampleRaw;
		default: return IrOpcode::Barrier;
	}
}

bool IsPredicatedStore(ValueOpcode opcode) {
	using enum IR::ValueOpcode;
	switch (opcode) {
		case WriteSharedU32:
		case WriteSharedU32x2:
		case WriteSharedU32x3:
		case WriteSharedU32x4:
		case StoreBufferU32:
		case StoreBufferU32x2:
		case StoreBufferU32x3:
		case StoreBufferU32x4: return true;
		default: return false;
	}
}

class ProofInput {
public:
	bool Populate(const IR::Program& source) {
		if (source.blocks.empty()) return false;
		program.SetWaveSize(source.wave_size);
		for (const auto* block: source.blocks) {
			auto& output = program.CreateBlock();
			program.BlockOrder().push_back(&output);
			blocks.emplace(block, &output);
			for (const auto& inst: *block) {
				auto& value = program.CreateValue(ProofOpcode(inst.GetOpcode()), ProofType(inst.GetType()), inst.Flags<uint64_t>());
				output.AppendInstruction(&value);
				values.emplace(&inst, &value);
			}
		}
		program.SetEntryBlock(*blocks.at(source.blocks.front()));
		for (const auto* block: source.blocks) {
			if (!AddControlFlow(*block)) {
				LOGF("merged hull whole-quad proof cannot represent control flow at %08x\n", block->start_pc);
				return false;
			}
			for (const auto& inst: *block) {
				if (!AddInputs(source, inst)) {
					LOGF("merged hull whole-quad proof cannot represent operands of %s at %08x\n",
				     std::string(IR::ValueOpcodeName(inst.GetOpcode())).c_str(), block->start_pc);
					return false;
				}
			}
		}
		for (const auto* block: source.blocks) {
			for (const auto& inst: *block) {
				if (!IsWaveCondition(inst)) continue;
				if (!wave_conditions.contains(&inst)) return false;
				for (const auto& use: inst.Uses()) {
					if (use.user->GetOpcode() == ValueOpcode::Reference) continue;
					if (!inverted_conditions.contains(use.user)) return false;
					for (const auto& next: use.user->Uses()) {
						if (next.user->GetOpcode() != ValueOpcode::Reference) return false;
					}
				}
			}
		}
		return true;
	}

	Native::IrValue* Convert(Value input) {
		input = input.Resolve();
		if (const auto* inst = input.TryInstruction()) return values.at(inst);
		if (!input.IsImmediate()) return nullptr;
		auto& result = program.CreateValue(IrOpcode::Void, ProofType(input.GetType()));
		switch (input.GetType()) {
			case IR::Type::U1: result.SetImmediateBool(input.U1()); break;
			case IR::Type::U8: result.SetImmediateU8(input.U8()); break;
			case IR::Type::U16: result.SetImmediateU16(input.U16()); break;
			case IR::Type::U32: result.SetImmediateU32(input.U32()); break;
			case IR::Type::U64: result.SetImmediateU64(input.U64()); break;
			case IR::Type::F16: result.SetImmediateF16Bits(input.F16Bits()); break;
			case IR::Type::F32: result.SetImmediateF32(input.F32Value()); break;
			case IR::Type::ScalarReg: result.SetRegister({Native::RegisterBank::Scalar, IR::RegIndex(input.ScalarRegister())}); break;
			case IR::Type::VectorReg: result.SetRegister({Native::RegisterBank::Vector, IR::RegIndex(input.VectorRegister())}); break;
			default: return nullptr;
		}
		return &result;
	}

	Native::IrProgram program;
	std::unordered_map<const Inst*, Native::IrValue*> values;
	std::vector<Native::QuadHelperGuard> guards;

private:
	static bool IsWaveCondition(const Inst& inst) {
		if (inst.GetOpcode() != ValueOpcode::ConditionRef) return false;
		const auto kind = inst.Flags<CFG::BranchCondition>();
		return kind == CFG::BranchCondition::ExecNonZero || kind == CFG::BranchCondition::VccNonZero;
	}

	bool AddControlFlow(const IR::Block& block) {
		auto& output = *blocks.at(&block);
		auto& meta = program.Metadata().blockInfo.emplace_back();
		meta.id = output.Id();
		meta.startPc = block.start_pc;
		meta.endPc = block.end_pc;
		meta.terminator.loopHeader = block.terminator.loop_header;
		for (const auto* next: block.ImmSuccessors()) output.AddBranch(blocks.at(next));
		using enum CFG::TerminatorKind;
		switch (block.terminator.kind) {
			case Return: return block.ImmSuccessors().empty();
			case Branch:
				meta.terminator.kind = Native::TerminatorKind::Branch;
				meta.terminator.trueBlock = blocks.at(block.terminator.true_block)->Id();
				return true;
			case ConditionalBranch: break;
			default: return false;
		}
		meta.terminator.kind = Native::TerminatorKind::ConditionalBranch;
		meta.terminator.trueBlock = blocks.at(block.terminator.true_block)->Id();
		meta.terminator.falseBlock = blocks.at(block.terminator.false_block)->Id();
		meta.terminator.condition = Native::BranchCondition::ScalarInstruction;
		auto condition = block.condition.Resolve();
		bool inverted = false;
		const auto* reference = condition.TryInstruction();
		if (reference && reference->GetOpcode() == ValueOpcode::LogicalNot) {
			inverted = true;
			reference = reference->Arg(0).Resolve().TryInstruction();
		}
		if (reference && reference->GetOpcode() == ValueOpcode::ConditionRef) {
			switch (reference->Flags<CFG::BranchCondition>()) {
				case CFG::BranchCondition::ExecNonZero:
					meta.terminator.condition = inverted ? Native::BranchCondition::ExecZero : Native::BranchCondition::ExecNonZero;
					condition = inverted ? block.condition.Resolve() : reference->Arg(0);
					break;
				case CFG::BranchCondition::VccNonZero:
					meta.terminator.condition = inverted ? Native::BranchCondition::VccZero : Native::BranchCondition::VccNonZero;
					condition = inverted ? block.condition.Resolve() : reference->Arg(0);
					break;
				default: break;
			}
			if (IsWaveCondition(*reference)) {
				wave_conditions.insert(reference);
				if (inverted) inverted_conditions.insert(block.condition.Resolve().TryInstruction());
			}
		}
		meta.condition = Convert(condition);
		return meta.condition != nullptr;
	}

	bool AddInputs(const IR::Program& source, const Inst& inst) {
		auto& output = *values.at(&inst);
		if (IsPredicatedStore(inst.GetOpcode())) {
			if (inst.NumArgs() == 0 || inst.Arg(inst.NumArgs() - 1u).Resolve().GetType() != IR::Type::U1) return false;
			guards.push_back({&output, inst.NumArgs() - 1u});
		}
		if (inst.GetOpcode() == ValueOpcode::ImageSampleRaw) {
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= source.memory_info.size()) return false;
			const auto flags = source.memory_info[index].image_sample_flags;
			auto& entries = program.Resources().memoryInfo;
			auto& info = entries.emplace_back();
			if ((flags & Decoder::ImageSampleFlagLevelZero) != 0) info.imageSampleFlags = Native::RdnaImageSampleFlagLevelZero;
			output.SetFlags(Native::MemoryFlags{static_cast<uint32_t>(entries.size() - 1)});
		}
		const bool reverse = inst.GetOpcode() == ValueOpcode::FPOrdGreaterThanEqual32 ||
		                     inst.GetOpcode() == ValueOpcode::FPOrdGreaterThan32;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			auto* value = Convert(inst.Arg(reverse ? 1 - index : index));
			if (value == nullptr) return false;
			if (inst.GetOpcode() == ValueOpcode::Phi) output.AddPhiOperand(blocks.at(inst.PhiBlock(index)), value);
			else output.AddArgument(value);
		}
		return true;
	}

	std::unordered_map<const IR::Block*, Native::IrBlock*> blocks;
	std::unordered_set<const Inst*> wave_conditions;
	std::unordered_set<const Inst*> inverted_conditions;
};

}

bool ProveHullQuadHelpersUnobserved(const IR::Program& program, const Inst& mask,
                                    std::span<Inst* const> reads, Value predicate) {
	try {
		ProofInput proof;
		if (!proof.Populate(program)) return false;
		const auto* own = proof.Convert(predicate);
		if (own == nullptr) return false;
		std::vector<Native::IrValue*> projected;
		for (const auto* read: reads) projected.push_back(proof.values.at(read));
		const auto refusal = Native::ProveQuadHelpersUnobserved(proof.program, *proof.values.at(&mask), projected, *own, proof.guards);
		if (!refusal.empty()) LOGF("merged hull whole-quad proof refused: %s\n", refusal.c_str());
		return refusal.empty();
	} catch (const std::exception& error) {
		LOGF("merged hull whole-quad proof could not represent the program: %s\n", error.what());
		return false;
	}
}

}
