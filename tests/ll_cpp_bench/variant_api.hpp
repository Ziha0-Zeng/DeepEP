#pragma once
#include "options.hpp"
#include "mpi_coordinator.hpp"
#include <acl/acl.h>

namespace llbench {
constexpr uint64_t workspaceBytes = 8ULL << 30;
// Borrowed resources. Only the runner initializes or releases MPI/ACL/SHMEM.
struct Runtime {
    Options options;
    const Coordinator* peers;
    aclrtStream stream;
    uint8_t* workspace;
    uint8_t* rdmaStage;
    uint64_t capacity, ffts;
    uint32_t cores, rdmaQps;
};
using RunVariant = void (*)(const Runtime*);
}  // namespace llbench
#define LL_BENCH_EXPORT extern "C" __attribute__((visibility("default")))
