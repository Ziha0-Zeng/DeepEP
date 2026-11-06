#pragma once
#include "../options.hpp"
#include "config.hpp"
#include <array>

namespace opsbench {
// Prepare the upstream Combine's input contract directly, outside the Event.
// Routes and identity-expert values match the existing DeepEP fixture.
struct Fixture {
    std::vector<int32_t> experts, assist, counts;
    std::vector<uint16_t> expanded, expected;
    std::vector<float> weights, expandedWeights;
    uint32_t rows = 0;
    Fixture(const llbench::Options& o, uint32_t bs)
    {
        const uint32_t perServer = std::min(o.world, 16U), servers = o.world / perServer;
        const uint32_t localExperts = o.experts / o.world;
        const uint32_t alignedBs = (bs + 7U) / 8U * 8U;
        experts = llbench::routes(o, bs);
        weights.assign(bs * o.topk, 1.0f / o.topk);
        expected.resize(bs * o.hidden);
        for (uint32_t t = 0; t < bs; ++t)
            for (uint32_t h = 0; h < o.hidden; ++h)
                expected[t * o.hidden + h] = llbench::encode(llbench::input_value(o.rank, t, h), o.dtype == "bf16");
        std::vector<std::vector<int32_t>> all(o.world);
        for (uint32_t source = 0; source < o.world; ++source) {
            auto peer = o;
            peer.rank = source;
            all[source] = llbench::routes(peer, bs);
        }
        // Cumulative counts use [local expert, source rank] order.
        for (uint32_t e = 0; e < localExperts; ++e) {
            for (uint32_t source = 0; source < o.world; ++source) {
                for (uint32_t t = 0; t < bs; ++t) {
                    for (uint32_t k = 0; k < o.topk; ++k) {
                        if (all[source][t * o.topk + k] != int32_t(o.rank * localExperts + e)) continue;
                        assist.insert(assist.end(), {int32_t(source), int32_t(t), int32_t(k)});
                        expandedWeights.push_back(1.0f / o.topk);
                        for (uint32_t h = 0; h < o.hidden; ++h)
                            expanded.push_back(llbench::encode(llbench::input_value(source, t, h), o.dtype == "bf16"));
                        ++rows;
                    }
                }
                counts.push_back(int32_t(rows));
            }
        }
        // Upstream appends each source-server's route masks after a * 3 int32s.
        // This PE gathers for sources with the same local rank within each server.
        const size_t maskBase = assist.size();
        assist.resize(maskBase + servers * (alignedBs + 8U), 0);
        for (uint32_t server = 0; server < servers; ++server) {
            const uint32_t source = server * perServer + o.rank % perServer;
            for (uint32_t t = 0; t < bs; ++t) {
                uint32_t mask = 0;
                for (uint32_t k = 0; k < o.topk; ++k) {
                    const uint32_t owner = uint32_t(all[source][t * o.topk + k]) / localExperts;
                    if (owner / perServer == o.rank / perServer) mask |= 1U << k;
                }
                assist[maskBase + server * (alignedBs + 8U) + 8U + t] = int32_t(mask);
            }
        }
    }
};
}  // namespace opsbench
