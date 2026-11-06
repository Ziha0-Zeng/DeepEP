#include "suite.hpp"
#include "variant_api.hpp"
#include "runtime/session.hpp"
#include "runtime/ll_core_config.hpp"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

namespace {
void check(int status, const char* operation)
{
    if (status) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
struct Module {
    void* handle = nullptr;
    llbench::RunVariant run = nullptr;
    explicit Module(const std::filesystem::path& path)
    {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error("Load " + path.string() + ": " + dlerror() + "; rebuild this variant");
        run = reinterpret_cast<llbench::RunVariant>(dlsym(handle, "ll_bench_run"));
        if (!run) {
            dlclose(handle);
            handle = nullptr;
            throw std::runtime_error("Missing ll_bench_run: rebuild " + path.string());
        }
    }
    // Keep kernel registrations alive until process exit, including error paths
    // where MPI_Abort must run before any in-flight device code is unloaded.
    ~Module() = default;
    Module(const Module&) = delete;
};
void marker(const std::filesystem::path& directory, int rank, const char* suffix, const char* message)
{
    std::ofstream file(directory / ("rank-" + std::to_string(rank) + suffix));
    file << message << '\n';
    file.close();
    if (!file) throw std::runtime_error("Cannot write result marker in " + directory.string());
}
}  // namespace

int run_suite(int argc, char** argv, const std::filesystem::path& executable)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    bool mpiActive = true;
    try {
        llbench::Coordinator peers;
        llbench::Suite suite(argc, argv, executable);
        auto options = suite.options(peers.rank, peers.world, peers.localRank);
        peers.verify(llbench::hash(options.signature() + suite.signature()));
        // Preload local groups once. Each module depends on a uniquely named
        // kernel library, so the base and experiment launch symbols stay separate.
        std::map<std::string, std::unique_ptr<Module>> modules;
        for (size_t i = 0; i < suite.variants.size(); ++i) {
            const auto path = suite.library(i).string();
            if (!modules.count(path)) modules.emplace(path, std::make_unique<Module>(path));
            const auto out = suite.output(options.output, i);
            if (std::filesystem::exists(out / ("rank-" + std::to_string(peers.rank) + ".csv")))
                throw std::runtime_error("Results already exist: " + out.string());
        }
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(options.device), "set device");
        aclrtStream stream = nullptr;
        check(aclrtCreateStream(&stream), "create stream");
        const auto cores = deepep::query_ll_core_config().active;
        setenv("DEEPEP_LL_BACKEND", "rdma", 1);
        {
            if (!peers.rank)
                std::cout << "Initializing shared MPI/ACL/SHMEM session once for " << suite.variants.size() << " phases"
                          << std::endl;
            deepep::Session session(peers.rank, peers.world, llbench::workspaceBytes, "", options.timeout, false, false,
                                    true);
            llbench::Runtime runtime{options,
                                     &peers,
                                     stream,
                                     static_cast<uint8_t*>(session.workspace()),
                                     static_cast<uint8_t*>(session.ll_rdma_stage()),
                                     session.capacity(),
                                     shmemx_get_ffts_config(),
                                     cores,
                                     session.ll_rdma_qps()};
            for (size_t i = 0; i < suite.variants.size(); ++i) {
                check(aclrtSynchronizeStream(stream), "phase completion before reset");
                peers.barrier();
                if (i) {
                    // Kernels retire their transport work before returning.
                    // All ranks have finished before any peer clears shared
                    // payloads, flags, bank generations and registered staging.
                    const auto bytes = session.workspace_allocation_bytes();
                    check(aclrtMemset(runtime.workspace, bytes, 0, bytes), "reset benchmark workspace");
                }
                peers.barrier();
                runtime.options.output = suite.output(options.output, i).string();
                if (!peers.rank)
                    std::cout << "phase=" << i << " variant=" << suite.variants[i]
                              << " output=" << runtime.options.output << std::endl;
                modules.at(suite.library(i).string())->run(&runtime);
                check(aclrtSynchronizeStream(stream), "phase completion");
                peers.barrier();
                marker(runtime.options.output, peers.rank, ".SAMPLED", "precision and sampling passed");
            }
            peers.barrier();
            session.close(true);
            peers.barrier();
            if (!peers.rank) std::cout << "Shared SHMEM session finalized once" << std::endl;
        }
        check(aclrtDestroyStream(stream), "destroy stream");
        check(aclrtResetDevice(options.device), "reset device context");
        check(aclFinalize(), "aclFinalize");
        peers.barrier();
        for (size_t i = 0; i < suite.variants.size(); ++i)
            marker(suite.output(options.output, i), peers.rank, ".PASS",
                   "precision, sampling and shared SHMEM finalize passed");
        llbench::mpi_check(MPI_Finalize(), "MPI_Finalize");
        mpiActive = false;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        if (mpiActive) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "LL Combine benchmark: one MPI/ACL/SHMEM initialization for all variants.\n"
                     "--variants base,opt --build-root DIR --output-dir DIR\n"
                     "--tokens 32,48,64,72,256 --hidden 3584 --topk 8 --experts 384 --capacity 256\n"
                     "--dtype bf16|fp16 --route random|hotspot --seed 0 --warmup 10 --iters 100\n"
                     "--devices LIST --entry-sync auto|host|device --profile 0|1|2 --timeout 300\n";
        return 0;
    }
    return run_suite(argc, argv, std::filesystem::canonical("/proc/self/exe"));
}
