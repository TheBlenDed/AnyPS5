#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/KeyMissObserver.hpp"
#include <cstdio>
#include <stdexcept>

using Observer = AgcDriver::DriverDetail::DispatchKeyMissObserver;
using Outcome = Observer::Outcome;

void Require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}

int main() {
    try {
        const auto source = std::make_shared<int>(1);
        const auto shader = std::make_shared<int>(2);
        const auto device = std::make_shared<int>(3);
        Observer::Observation base;
        base.source = source;
        base.shader = shader;
        base.device = device;
        base.pointers = {reinterpret_cast<std::uintptr_t>(source.get()), reinterpret_cast<std::uintptr_t>(shader.get()), reinterpret_cast<std::uintptr_t>(device.get())};
        base.fixed[0] = 0x1000;
        base.fixed[15] = 4;
        base.words = {10, 20, 30, 40};
        base.wordCount = 4;
        base.exactKey = 100;
        base.mask = Observer::DependencyMask(1, 0, 4, true);
        const auto pair = [&](const Observer::Observation& first, const Observer::Observation& second) {
            Observer observer;
            observer.Observe(first);
            return observer.Observe(second);
        };
        Require(pair(base, base) == Outcome::Unchanged, "unchanged words were blamed on key churn");
        auto unused = base;
        unused.words[3]++;
        unused.exactKey++;
        Require(pair(base, unused) == Outcome::Potential, "unused-word change was not classified");
        unused.exactKey = base.exactKey;
        Require(pair(base, unused) == Outcome::Conservative, "same-key collision was blamed on key churn");
        for (std::size_t used = 0; used < base.wordCount; ++used) {
            auto previous = base;
            previous.mask = std::uint64_t{1} << used;
            auto current = previous;
            current.words[used]++;
            current.exactKey++;
            Require(pair(previous, current) == Outcome::Used, "shader data, pointer or branch dependency was ignored");
        }
        auto changedMask = base;
        changedMask.mask = 8;
        changedMask.words[3]++;
        changedMask.exactKey++;
        Require(pair(base, changedMask) == Outcome::Used, "new observation dependency was ignored");
        Require(pair(changedMask, base) == Outcome::Used, "old observation dependency was ignored");
        auto unknown = base;
        unknown.mask.reset();
        Require(pair(base, unknown) == Outcome::Conservative && pair(unknown, base) == Outcome::Conservative, "unknown dependency was treated as unused");
        Require(!Observer::DependencyMask(0, 1, 4, true) && !Observer::DependencyMask(1, ~std::uint64_t{0}, 4, true) &&
            !Observer::DependencyMask(1, 0, 4, false) && !Observer::DependencyMask(1, 0, 33, true) &&
            !Observer::DependencyMask(1, 0, 0, true) && Observer::DependencyMask(1ull << 31u, 0, 32, true), "dependency mask conservatism or bounds failed");
        for (std::size_t field = 0; field < base.fixed.size(); ++field) {
            auto changed = base;
            changed.fixed[field]++;
            Require(pair(base, changed) == Outcome::NoPrevious, "different fixed dispatch identity paired");
        }
        auto differentCount = base;
        differentCount.wordCount = 3;
        Require(pair(base, differentCount) == Outcome::NoPrevious, "different input count paired");
        const auto linked = std::make_shared<int>(4);
        auto changedSource = base;
        changedSource.source = linked;
        changedSource.pointers[0] = reinterpret_cast<std::uintptr_t>(linked.get());
        Require(pair(base, changedSource) == Outcome::NoPrevious, "linked or converted normalized source paired");
        auto changedOwner = base;
        changedOwner.source = std::shared_ptr<const void>(linked, source.get());
        Require(pair(base, changedOwner) == Outcome::NoPrevious, "reused pointer with different source owner paired");
        changedOwner = base;
        changedOwner.shader = std::shared_ptr<const void>(linked, shader.get());
        Require(pair(base, changedOwner) == Outcome::NoPrevious, "different original shader paired");
        changedOwner = base;
        changedOwner.device = std::shared_ptr<const void>(linked, device.get());
        Require(pair(base, changedOwner) == Outcome::NoPrevious, "different device paired");
        Observer bounded;
        for (std::size_t i = 0; i < 1024; ++i) {
            auto unique = base;
            unique.fixed[0] += i * 256;
            Require(bounded.Observe(unique) != Outcome::Potential, "slot collision classified as key churn");
        }
        Require(bounded.Size() <= Observer::Capacity && bounded.Totals().overwritten != 0, "observation table was not bounded");
        std::uint64_t classified = 0;
        for (const auto count : bounded.Totals().outcomes) classified += count;
        Require(classified == bounded.Totals().total && classified == 1024, "categories do not partition eligible observations");
        auto transient = std::make_shared<int>(5);
        auto expired = base;
        expired.source = transient;
        expired.pointers[0] = reinterpret_cast<std::uintptr_t>(transient.get());
        bounded.Observe(expired);
        transient.reset();
        Require(expired.source.expired() && bounded.Observe(expired) == Outcome::Conservative, "observer retained or reused expired source");
        auto oversized = base;
        oversized.wordCount = Observer::MaximumWords + 1;
        Require(bounded.Observe(oversized) == Outcome::Conservative, "oversized word list was indexed");
        Require(Observer::Eligible(true, 0, false, false, false, false, false), "real direct exact miss excluded");
        Require(!Observer::Eligible(false, 0, false, false, false, false, false) &&
            !Observer::Eligible(true, 0x51, false, false, false, false, false) &&
            !Observer::Eligible(true, 0, true, false, false, false, false) &&
            !Observer::Eligible(true, 0, false, true, false, false, false) &&
            !Observer::Eligible(true, 0, false, false, true, false, false) &&
            !Observer::Eligible(true, 0, false, false, false, true, false) &&
            !Observer::Eligible(true, 0, false, false, false, false, true), "non-miss, indirect, ahead, resolving or OFF input accepted");
        std::puts("dispatch exact-key miss observer tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
