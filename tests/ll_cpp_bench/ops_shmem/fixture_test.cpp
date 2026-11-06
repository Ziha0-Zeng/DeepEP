#include "fixture.hpp"
#include <cassert>
#include <set>
#include <tuple>
#include <iostream>

int main()
{
    for (uint32_t world : {2U, 4U, 8U, 16U, 32U}) {
        for (uint32_t bs : {32U, 48U, 64U, 72U, 256U}) {
            for (const auto& route : {"random", "hotspot"}) {
                llbench::Options o;
                o.world = world;
                o.hidden = 16;
                o.route = route;
                const auto group = std::min(world, 16U), servers = world / group;
                const auto alignedBs = (bs + 7) / 8 * 8;
                std::vector<std::vector<int32_t>> ids(world);
                for (uint32_t rank = 0; rank < world; ++rank) {
                    auto peer = o;
                    peer.rank = rank;
                    ids[rank] = llbench::routes(peer, bs);
                }
                std::vector<std::vector<uint32_t>> observed(world, std::vector<uint32_t>(servers * alignedBs, 0));
                std::set<std::tuple<uint32_t, uint32_t, uint32_t>> seen;
                for (uint32_t rank = 0; rank < world; ++rank) {
                    o.rank = rank;
                    opsbench::Fixture f(o, bs);
                    assert(f.counts.back() == int32_t(f.rows));
                    assert(std::is_sorted(f.counts.begin(), f.counts.end()));
                    for (uint32_t row = 0; row < f.rows; ++row) {
                        const auto source = uint32_t(f.assist[row * 3]);
                        const auto token = uint32_t(f.assist[row * 3 + 1]);
                        const auto k = uint32_t(f.assist[row * 3 + 2]);
                        assert(uint32_t(ids[source][token * o.topk + k]) / (o.experts / world) == rank);
                        assert(seen.emplace(source, token, k).second);
                        assert(f.expandedWeights[row] == 1.0f / o.topk);
                        for (uint32_t h = 0; h < o.hidden; ++h)
                            assert(f.expanded[row * o.hidden + h] ==
                                   llbench::encode(llbench::input_value(source, token, h), true));
                        const auto forward = rank / group * group + source % group;
                        observed[forward][source / group * alignedBs + token] |= 1U << k;
                    }
                }
                assert(seen.size() == size_t(world) * bs * o.topk);
                for (uint32_t rank = 0; rank < world; ++rank) {
                    o.rank = rank;
                    opsbench::Fixture f(o, bs);
                    for (uint32_t s = 0; s < servers; ++s)
                        for (uint32_t t = 0; t < alignedBs; ++t)
                            assert(uint32_t(f.assist[f.rows * 3 + s * (alignedBs + 8) + 8 + t]) ==
                                   observed[rank][s * alignedBs + t]);
                }
            }
        }
    }
    std::cout << "ops fixture: every route delivered exactly once; gather masks match scatter slots\n";
}
