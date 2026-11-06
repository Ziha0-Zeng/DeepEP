#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llbench {
inline uint32_t number(const std::string& text, uint32_t low, uint32_t high)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected unsigned integer: " + text);
    auto value = std::stoull(text);
    if (value < low || value > high) throw std::invalid_argument("Out-of-range value: " + text);
    return uint32_t(value);
}
struct Options {
    uint32_t rank = 0, world = 16, device = 0;
    uint32_t hidden = 3584, topk = 8, experts = 384, capacity = 256, warmup = 10, iterations = 100, seed = 0,
             timeout = 300, profile = 0;
    std::vector<uint32_t> tokens{32, 48, 64, 72, 256};
    std::string output = "results/ll-cpp", dtype = "bf16", route = "random", entrySync = "auto";
    static Options parse(int argc, char** argv, uint32_t rank = 0, uint32_t world = 16, uint32_t localRank = 0)
    {
        Options o;
        o.rank = rank;
        o.world = world;
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc || std::string(argv[i]).rfind("--", 0) != 0 || !args.emplace(argv[i], argv[i + 1]).second)
                throw std::invalid_argument("Expected unique --option value pairs");
        }
        auto text = [&](const char* key, std::string& value) {
            auto i = args.find(key);
            if (i != args.end()) {
                value = i->second;
                args.erase(i);
            }
        };
        auto integer = [&](const char* key, uint32_t& value, uint32_t low, uint32_t high) {
            auto i = args.find(key);
            if (i != args.end()) {
                value = number(i->second, low, high);
                args.erase(i);
            }
        };
        text("--output-dir", o.output);
        text("--dtype", o.dtype);
        text("--route", o.route);
        text("--entry-sync", o.entrySync);
        if (o.entrySync == "auto") o.entrySync = world <= 16 ? "device" : "host";
        if ((o.entrySync != "host" && o.entrySync != "device") || (world > 16 && o.entrySync == "device"))
            throw std::invalid_argument("entry-sync supports device for single-host EP, or host for all EP sizes");
        const uint32_t perHost = std::min(world, 16U);
        std::string devices;
        for (uint32_t i = 0; i < perHost; ++i) devices += (i ? "," : "") + std::to_string(i);
        text("--devices", devices);
        std::vector<uint32_t> ids;
        std::stringstream deviceStream(devices);
        std::string part;
        while (std::getline(deviceStream, part, ',')) ids.push_back(number(part, 0, 255));
        auto sorted = ids;
        std::sort(sorted.begin(), sorted.end());
        if (devices.empty() || devices.back() == ',' || ids.size() != perHost || localRank >= perHost ||
            std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::invalid_argument("devices must list one distinct device ID per local MPI rank");
        o.device = ids[localRank];
        integer("--hidden", o.hidden, 16, 7168);
        integer("--topk", o.topk, 1, 32);
        integer("--experts", o.experts, 16, 1024);
        integer("--capacity", o.capacity, 1, 256);
        integer("--warmup", o.warmup, 0, 10000);
        integer("--iters", o.iterations, 1, 10000);
        integer("--seed", o.seed, 0, 0xffffffffU);
        integer("--timeout", o.timeout, 1, 3600);
        integer("--profile", o.profile, 0, 2);
        if (auto i = args.find("--tokens"); i != args.end()) {
            if (i->second.empty() || i->second.back() == ',') throw std::invalid_argument("Empty token count");
            o.tokens.clear();
            std::stringstream stream(i->second);
            std::string part;
            while (std::getline(stream, part, ',')) o.tokens.push_back(number(part, 1, 256));
            args.erase(i);
        }
        if (!args.empty()) throw std::invalid_argument("Unknown option: " + args.begin()->first);
        if ((o.world != 2 && o.world != 4 && o.world != 8 && o.world != 16 && o.world != 32) || o.rank >= o.world ||
            o.output.empty() || o.tokens.empty() || o.hidden % 16 || o.experts % o.world || o.topk > o.experts ||
            (o.dtype != "bf16" && o.dtype != "fp16") || (o.route != "random" && o.route != "hotspot"))
            throw std::invalid_argument("Invalid topology, shape, dtype or routing policy");
        if (*std::max_element(o.tokens.begin(), o.tokens.end()) > o.capacity)
            throw std::invalid_argument("capacity must cover every token count");
        return o;
    }
    std::string signature() const
    {
        std::ostringstream s;
        s << world << '/' << hidden << '/' << topk << '/' << experts << '/' << capacity << '/' << warmup << '/'
          << iterations << '/' << seed << '/' << dtype << '/' << route << '/' << profile << '/' << entrySync;
        for (auto b : tokens) s << '/' << b;
        return s.str();
    }
};
inline uint64_t hash(const std::string& text)
{
    uint64_t value = 14695981039346656037ULL;
    for (unsigned char c : text) {
        value ^= c;
        value *= 1099511628211ULL;
    }
    return value;
}
// Exactly representable input values; every token/rank has a different pattern.
inline float input_value(uint32_t rank, uint32_t token, uint32_t column)
{
    return float(int((rank * 37U + token * 13U + column * 3U) % 127U) - 63) / 32.0f;
}
inline uint16_t encode(float value, bool bf16)
{
    uint32_t word;
    std::memcpy(&word, &value, 4);
    if (bf16) return uint16_t((word + 0x7fffU + ((word >> 16) & 1U)) >> 16);
    // Test inputs/results are finite normal FP16 values or zero.
    uint32_t sign = (word >> 16) & 0x8000U, exp = (word >> 23) & 0xffU, mantissa = word & 0x7fffffU;
    if (exp == 0) return uint16_t(sign);
    if (exp < 113 || exp > 142) throw std::runtime_error("FP16 fixture conversion outside normal range");
    uint32_t rounded = mantissa + 0xfffU + ((mantissa >> 13) & 1U);
    return uint16_t(sign + ((exp - 112) << 10) + (rounded >> 13));
}
inline float decode(uint16_t value, bool bf16)
{
    uint32_t word;
    if (bf16)
        word = uint32_t(value) << 16;
    else {
        uint32_t exp = (value >> 10) & 31U, mantissa = value & 1023U;
        if (exp == 0) return (value & 0x8000U ? -1.0f : 1.0f) * std::ldexp(float(mantissa), -24);
        word = (uint32_t(value & 0x8000U) << 16) | ((exp == 31 ? 255U : exp + 112U) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &word, 4);
    return result;
}
inline std::vector<int32_t> routes(const Options& o, uint32_t bs)
{
    std::mt19937 rng(o.seed + o.rank * 1009U);
    std::vector<int32_t> result(bs * o.topk), experts(o.experts);
    std::iota(experts.begin(), experts.end(), 0);
    for (uint32_t t = 0; t < bs; ++t) {
        if (o.route == "random") std::shuffle(experts.begin(), experts.end(), rng);
        for (uint32_t k = 0; k < o.topk; ++k) result[t * o.topk + k] = o.route == "random" ? experts[k] : int32_t(k);
    }
    return result;
}
struct Stats {
    double mean, min, max, p50, p99;
};
inline Stats stats(std::vector<float> values)
{
    if (values.empty()) throw std::invalid_argument("No samples");
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    return {mean, values.front(), values.back(), values[(values.size() - 1) / 2],
            values[size_t(std::ceil(values.size() * .99)) - 1]};
}
}  // namespace llbench
