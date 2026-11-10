#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

int main(int argc, char** argv) {
    try {
        using namespace AgcDriver;
        using DriverDetail::DrawPipeline;
        using DriverDetail::DrawFinisher;
        if (DrawPipeline::Depth() == 0) return 77;
        const auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
        const bool disabled = mode == "disabled" || mode == "finisher-disabled";
        if (mode == "links") {
            if (!PerformanceTrace::Configured()) throw std::runtime_error("trace is not configured for the link check");
            PerformanceTrace::Present(1, 1, nullptr);
            auto& pipeline = DrawPipeline::Queue0();
            const auto depth = DrawPipeline::Depth();
            std::atomic<bool> go{false};
            std::atomic<bool> producerDone{false};
            pipeline.Enqueue([&] { TraceBatchTag = 42; while (!go.load(std::memory_order_relaxed)) std::this_thread::yield(); }, {}, 0, {}, {}, {}, false, {5, 0, 1, 0, true});
            pipeline.Enqueue([&] { while (!producerDone.load(std::memory_order_relaxed)) std::this_thread::yield(); }, {}, 0, {}, {}, {}, false, {6, 0, 2, 0, false});
            for (std::size_t item = 2; item < depth; ++item) pipeline.Enqueue([] {}, {}, 0, {}, {}, {}, false, {6, 0, 2, 0, false});
            std::atomic<bool> waiting{false};
            std::thread producer([&] {
                PerformanceTrace::BindThread("links producer");
                waiting.store(true, std::memory_order_relaxed);
                pipeline.Enqueue([] {}, {}, 0, {}, {}, {}, false, {7, 0, 3, 0, true});
                producerDone.store(true, std::memory_order_relaxed);
            });
            while (!waiting.load(std::memory_order_relaxed)) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            go.store(true, std::memory_order_relaxed);
            producer.join();
            pipeline.Drain(DrawPipeline::DrainReason::Submission);
            DrawPipeline::TraceFlip(DrawPipeline::TraceIdentity{11, 0, 44, 0, false}, 3, 2);
            std::uint64_t dropped = 0;
            const auto links = DrawPipeline::CollectTraceLinks(dropped);
            if (links.size() != 2 || dropped != 0) throw std::runtime_error("expected one room-wait link and one flip link");
            const auto& wait = links[0];
            if (wait.flip) throw std::runtime_error("the room-wait link is a flip link");
            if (wait.freedItem != 1 || wait.freedAt < wait.at) throw std::runtime_error("the room-wait link named the wrong freeing item");
            if (wait.freed.submission != 5 || wait.freed.queue != 0 || wait.freed.offset != 1 || !wait.freed.draw) throw std::runtime_error("the room-wait link named the wrong freeing identity");
            if (wait.pending.submission != 7 || wait.pending.queue != 0 || wait.pending.offset != 3 || !wait.pending.draw) throw std::runtime_error("the room-wait link named the wrong pending identity");
            if (wait.batch != 42) throw std::runtime_error("the room-wait link lost the freeing commit's batch tag (batch=" + std::to_string(wait.batch) + ")");
            if (wait.thread == nullptr) throw std::runtime_error("the room-wait link lost its producer thread");
            const auto& flip = links[1];
            if (!flip.flip || flip.pending.submission != 11 || flip.pending.queue != 0 || flip.pending.offset != 44 || flip.batch != 3 || flip.unsignaled != 2 || flip.thread == nullptr) throw std::runtime_error("the flip link lost its identity");
            std::atomic<bool> blocked{false};
            pipeline.Enqueue([&] { TraceBatchTag = 43; while (!blocked.load(std::memory_order_relaxed)) std::this_thread::yield(); }, {}, 0, {}, {}, {}, false, {8, 0, 5, 0, true});
            for (std::size_t item = 1; item < depth; ++item) pipeline.Enqueue([] {}, {}, 0, {}, {}, {}, false, {9, 0, 6, 0, false});
            std::atomic<unsigned> arrived{0};
            const auto produce = [&](std::size_t offset) {
                PerformanceTrace::BindThread("links producer");
                arrived.fetch_add(1, std::memory_order_relaxed);
                pipeline.Enqueue([] {}, {}, 0, {}, {}, {}, false, {10, 0, offset, 0, true});
            };
            std::thread first([&] { produce(7); });
            std::thread second([&] { produce(8); });
            while (arrived.load(std::memory_order_relaxed) < 2) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            blocked.store(true, std::memory_order_relaxed);
            first.join();
            second.join();
            pipeline.Drain(DrawPipeline::DrainReason::Submission);
            dropped = 0;
            const auto more = DrawPipeline::CollectTraceLinks(dropped);
            if (more.size() != 2 || dropped != 0) throw std::runtime_error("expected two room-wait links from two producers");
            const auto& low = more[0].freedItem <= more[1].freedItem ? more[0] : more[1];
            const auto& high = more[0].freedItem <= more[1].freedItem ? more[1] : more[0];
            if (low.freedItem != depth + 2 || high.freedItem != 2 * depth + 2 || low.freed.submission != 8 || high.freed.submission != 10 || !low.freed.draw || !high.freed.draw || low.batch != 43 || high.batch != 43 || low.pending.offset + high.pending.offset != 15 || low.thread == nullptr || high.thread == nullptr) throw std::runtime_error("the simultaneous room-wait links were not stamped per waiter");
            std::cout << "draw pipeline trace links checked\n";
            return 0;
        }
        const bool finisher = mode.starts_with("finisher");
        const char* worker = finisher ? "draw finisher" : "draw committer";
        if (finisher && !DrawFinisher::Enabled()) throw std::runtime_error("draw finisher is disabled");
        if (PerformanceTrace::Configured() == disabled) throw std::runtime_error("unexpected trace configuration");
        if (!disabled) PerformanceTrace::Present(1, 1, nullptr);
        bool checked = false;
        const char* failure = nullptr;
        const auto checkWorker = [&] {
            auto* trace = PerformanceTrace::Thread();
            if (disabled) {
                if (trace != nullptr) { failure = "unconfigured draw worker allocated a trace buffer"; return; }
                checked = true;
                return;
            }
            if (trace == nullptr) { failure = "draw worker CPU spans are absent from the trace"; return; }
            {
                PerformanceTimer timer("DrawPipeline.TraceRegression");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto snapshot = trace->Collect();
            if (snapshot.name != worker || snapshot.dropped) { failure = "draw worker trace identity or capacity is incorrect"; return; }
            const auto found = [](const auto& event) { return event.scope != nullptr && std::string_view(event.scope) == "DrawPipeline.TraceRegression" && event.stage == nullptr && event.length > std::chrono::steady_clock::duration::zero(); };
            if (std::none_of(snapshot.slices.begin(), snapshot.slices.end(), found) || std::none_of(snapshot.totals.begin(), snapshot.totals.end(), found))
                { failure = "draw work was missing from trace slices or totals"; return; }
            checked = true;
        };
        if (finisher) {
            DrawFinisher::Get().Enqueue(checkWorker);
            DrawFinisher::Get().Drain();
        } else {
            DrawPipeline::Queue0().Enqueue(checkWorker, {});
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
        }
        if (failure != nullptr) throw std::runtime_error(failure);
        if (!checked) throw std::runtime_error("draw worker did not run the trace check");
        std::cout << worker << " tracing checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
