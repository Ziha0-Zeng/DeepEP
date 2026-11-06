#pragma once
#include <cstdint>

namespace opsbench {
constexpr uint64_t workspaceBytes = 8ULL << 30;
constexpr uint64_t windowBytes = 4ULL << 30;
constexpr uint64_t stateBytes = 1ULL << 20;
// Layout of the pinned kernel's context. These are SHMEM mappings / PE IDs;
// no HCCL context or channel is created by this benchmark.
struct Context {
    uint32_t rank, ranksPerServer;
    uint64_t unusedContext;
    uint64_t localWindows[1024];
    uint64_t peers[1024];
    uint32_t channelsPerRank;
    bool unusedGroupSend;
    uint64_t remoteWindows[1024];
    uint64_t remoteWindowBytes;
};
struct Config {
    uint32_t rank, world, bs, hidden, topk, experts, rows, cores;
    uint64_t ubBytes, ffts;
};
struct Args {
    uint8_t *context, *expanded, *experts, *assist, *counts, *weights, *expandedWeights, *output, *config;
    uint32_t cores, bf16;
};
}  // namespace opsbench
extern "C" void ops_shmem_combine_do(void* stream, const opsbench::Args* args);
