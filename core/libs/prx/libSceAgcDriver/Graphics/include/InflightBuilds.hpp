#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INFLIGHTBUILDS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INFLIGHTBUILDS_HPP

#include <cstddef>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace AgcDriver::Graphics {

class InflightBuilds {
public:
    using Key = std::vector<std::byte>;
    struct Build {
        std::shared_future<void> ready;
    };
    struct Claim {
        bool owner = false;
        std::shared_ptr<Build> build;
    };

    Claim Acquire(const Key& key, const std::shared_future<void>& ready) {
        std::lock_guard lock(mutex);
        if (const auto found = builds.find(key); found != builds.end()) return {false, found->second};
        auto build = std::make_shared<Build>(Build{ready});
        builds.emplace(key, build);
        return {true, std::move(build)};
    }

    std::shared_future<void> Find(const Key& key) const {
        std::lock_guard lock(mutex);
        const auto found = builds.find(key);
        return found != builds.end() ? found->second->ready : std::shared_future<void>{};
    }

    void Release(const Key& key, const std::shared_ptr<Build>& build) {
        std::lock_guard lock(mutex);
        const auto found = builds.find(key);
        if (found != builds.end() && found->second == build) builds.erase(found);
    }

    std::size_t Size() const {
        std::lock_guard lock(mutex);
        return builds.size();
    }

private:
    mutable std::mutex mutex;
    std::map<Key, std::shared_ptr<Build>> builds;
};

}

#endif
