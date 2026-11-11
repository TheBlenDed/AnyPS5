#include "prx/libSceAgcDriver/Graphics/include/InflightBuilds.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

using AgcDriver::Graphics::InflightBuilds;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        InflightBuilds builds;
        const InflightBuilds::Key firstKey{std::byte{1}, std::byte{2}};
        const InflightBuilds::Key otherKey{std::byte{1}, std::byte{3}};
        std::promise<void> first;
        std::promise<void> other;
        std::promise<void> replacement;
        const auto owner = builds.Acquire(firstKey, first.get_future().share());
        const auto duplicate = builds.Acquire(firstKey, other.get_future().share());
        Require(owner.owner && !duplicate.owner && duplicate.build == owner.build, "duplicate key did not attach");
        Require(duplicate.build->ready.wait_for(std::chrono::seconds(0)) == std::future_status::timeout, "completion became ready early");
        first.set_value();
        Require(duplicate.build->ready.wait_for(std::chrono::seconds(0)) == std::future_status::ready, "owner did not complete its attached future");
        const auto distinct = builds.Acquire(otherKey, duplicate.build->ready);
        Require(distinct.owner && builds.Size() == 2, "full keys were combined");
        builds.Release(firstKey, owner.build);
        const auto current = builds.Acquire(firstKey, replacement.get_future().share());
        builds.Release(firstKey, owner.build);
        Require(current.owner && builds.Find(firstKey).valid(), "late owner release erased a replacement");
        builds.Release(otherKey, current.build);
        Require(builds.Size() == 2, "mismatched owner released a different key");
        builds.Release(firstKey, current.build);
        builds.Release(otherKey, distinct.build);
        Require(builds.Size() == 0, "keys were not released");
        std::promise<void> finished;
        const auto pending = finished.get_future().share();
        std::atomic<unsigned> owners{0};
        std::atomic<unsigned> attached{0};
        std::vector<std::thread> threads;
        for (unsigned i = 0; i < 16; ++i) {
            threads.emplace_back([&] {
                const auto claim = builds.Acquire(firstKey, pending);
                (claim.owner ? owners : attached).fetch_add(1);
            });
        }
        for (auto& thread : threads) thread.join();
        Require(owners.load() == 1 && attached.load() == 15 && builds.Size() == 1, "contention selected multiple owners");
        std::puts("inflight pipeline builds passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "inflight pipeline builds failed: %s\n", error.what());
        return 1;
    }
}
