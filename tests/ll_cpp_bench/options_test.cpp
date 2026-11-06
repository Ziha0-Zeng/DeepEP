#include "options.hpp"
#include <cassert>
#include <set>

llbench::Options parse(std::vector<std::string> args)
{
    std::vector<char*> argv;
    for (auto& s : args) argv.push_back(s.data());
    return llbench::Options::parse(int(argv.size()), argv.data(), 17, 32, 1);
}
int main()
{
    for (uint32_t world : {2U, 4U, 8U, 16U}) {
        char name[] = "bench";
        char* argv[] = {name};
        auto small = llbench::Options::parse(1, argv, world - 1, world, world - 1);
        assert(small.device == world - 1 && small.world == world);
        assert(small.entrySync == "device" && small.tokens == std::vector<uint32_t>({32, 48, 64, 72, 256}));
    }
    auto o = parse({"bench", "--tokens", "1,72", "--dtype", "fp16"});
    assert(o.profile == 0);
    assert(o.entrySync == "host");
    auto diagnostic = parse({"bench", "--tokens", "1,72", "--dtype", "fp16", "--profile", "2"});
    assert(diagnostic.profile == 2 && diagnostic.signature() != o.signature());
    assert(o.rank == 17 && o.world == 32 && o.device == 1 && o.tokens.back() == 72);
    for (auto args : std::vector<std::vector<std::string>>{{"bench", "--tokens", "1,"},
                                                           {"bench", "--tokens", "0"},
                                                           {"bench", "--devices", "0,1"},
                                                           {"bench", "--iters", "-1"},
                                                           {"bench", "--profile", "3"},
                                                           {"bench", "--entry-sync", "device"},
                                                           {"bench", "--entry-sync", "unknown"},
                                                           {"bench", "--hidden", "17"},
                                                           {"bench", "--capacity", "4"},
                                                           {"bench", "--iters", "3", "--iters", "4"},
                                                           {"bench", "--unknown", "1"}}) {
        bool rejected = false;
        try {
            parse(args);
        } catch (const std::exception&) {
            rejected = true;
        }
        assert(rejected);
    }
    for (bool bf16 : {false, true})
        for (uint32_t rank = 0; rank < 32; ++rank)
            for (uint32_t t = 0; t < 72; ++t)
                for (uint32_t h = 0; h < 128; ++h) {
                    float value = llbench::input_value(rank, t, h);
                    assert(llbench::decode(llbench::encode(value, bf16), bf16) == value);
                }
    assert(std::isnan(llbench::decode(0x7e00, false)));
    assert(std::isinf(llbench::decode(0x7c00, false)));
    auto random = llbench::routes(o, 72);
    assert(random == llbench::routes(o, 72));
    for (uint32_t t = 0; t < 72; ++t) {
        std::set<int> unique(random.begin() + t * o.topk, random.begin() + (t + 1) * o.topk);
        assert(unique.size() == o.topk && *unique.begin() >= 0 && *unique.rbegin() < int(o.experts));
    }
    o.route = "hotspot";
    auto hot = llbench::routes(o, 72);
    for (size_t i = 0; i < hot.size(); ++i) assert(hot[i] == int(i % o.topk));
    auto s = llbench::stats({4, 1, 3, 2});
    assert(s.mean == 2.5 && s.min == 1 && s.max == 4 && s.p50 == 2 && s.p99 == 4);
}
