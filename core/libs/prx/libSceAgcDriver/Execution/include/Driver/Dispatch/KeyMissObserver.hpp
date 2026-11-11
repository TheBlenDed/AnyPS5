#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_KEYMISSOBSERVER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_KEYMISSOBSERVER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace AgcDriver::DriverDetail {

class DispatchKeyMissObserver {
public:
    static constexpr std::size_t Capacity = 64;
    static constexpr std::size_t MaximumWords = 32;
    enum class Outcome : std::size_t { NoPrevious, Conservative, Unchanged, Used, Potential, Count };
    struct Observation {
        std::weak_ptr<const void> source;
        std::weak_ptr<const void> shader;
        std::weak_ptr<const void> device;
        std::array<std::uintptr_t, 3> pointers{};
        std::array<std::uint64_t, 16> fixed{};
        std::array<std::uint32_t, MaximumWords> words{};
        std::size_t wordCount = 0;
        std::uint64_t exactKey = 0;
        std::optional<std::uint64_t> mask;
    };
    struct Counters {
        std::uint64_t total = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(Outcome::Count)> outcomes{};
        std::uint64_t overwritten = 0;
    };

    static bool Eligible(bool enabled, std::uint32_t queue, bool indirect, bool resolving, bool ahead, bool cacheDisabled, bool validated) {
        return enabled && queue == 0 && !indirect && !resolving && !ahead && !cacheDisabled && !validated;
    }

    static std::optional<std::uint64_t> DependencyMask(std::uint64_t walked, std::uint64_t shaderReads, std::size_t count, bool complete) {
        if (!complete || walked == 0 || count == 0 || count > MaximumWords) return std::nullopt;
        const auto mask = walked | shaderReads;
        if ((mask >> count) != 0) return std::nullopt;
        return mask;
    }

    Outcome Observe(const Observation& observation) {
        ++counters.total;
        auto outcome = Outcome::Conservative;
        if (observation.wordCount <= MaximumWords && !observation.source.expired() && !observation.shader.expired() && !observation.device.expired()) {
            auto& slot = slots[Index(observation)];
            const auto& previous = slot.observation;
            const bool paired = slot.occupied && previous.pointers == observation.pointers && previous.fixed == observation.fixed &&
                previous.wordCount == observation.wordCount && SameOwner(previous.source, observation.source) &&
                SameOwner(previous.shader, observation.shader) && SameOwner(previous.device, observation.device);
            if (!paired) {
                outcome = observation.mask ? Outcome::NoPrevious : Outcome::Conservative;
                if (slot.occupied) ++counters.overwritten;
            } else if (previous.mask && observation.mask) {
                std::uint64_t changed = 0;
                for (std::size_t i = 0; i < observation.wordCount; ++i) {
                    if (previous.words[i] != observation.words[i]) changed |= std::uint64_t{1} << i;
                }
                if (changed == 0) outcome = Outcome::Unchanged;
                else if ((changed & (*previous.mask | *observation.mask)) != 0) outcome = Outcome::Used;
                else if (previous.exactKey != observation.exactKey) outcome = Outcome::Potential;
            }
            slot.observation = observation;
            slot.occupied = true;
        }
        ++counters.outcomes[static_cast<std::size_t>(outcome)];
        return outcome;
    }

    const Counters& Totals() const { return counters; }
    std::size_t Size() const {
        std::size_t count = 0;
        for (const auto& slot : slots) count += slot.occupied ? 1u : 0u;
        return count;
    }

private:
    struct Slot {
        Observation observation;
        bool occupied = false;
    };
    static bool SameOwner(const std::weak_ptr<const void>& left, const std::weak_ptr<const void>& right) {
        return !left.expired() && !right.expired() && !left.owner_before(right) && !right.owner_before(left);
    }
    static std::size_t Index(const Observation& observation) {
        auto value = observation.fixed[0];
        for (const auto pointer : observation.pointers) value = (value ^ static_cast<std::uint64_t>(pointer)) * 0x9e3779b97f4a7c15ull;
        return static_cast<std::size_t>((value ^ (value >> 32u)) & (Capacity - 1u));
    }
    std::array<Slot, Capacity> slots{};
    Counters counters;
};

}

#endif
