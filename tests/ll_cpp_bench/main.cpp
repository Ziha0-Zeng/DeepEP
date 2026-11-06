#include "options.hpp"
#include "profile.hpp"
#include "mpi_coordinator.hpp"
#include <cstdlib>
#include <acl/acl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include "variant_api.hpp"
#include "runtime/ll_core_config.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "kernels/dispatch/low_latency/common/ll_rdma_layout.h"
#include "kernels/dispatch/low_latency/common/comm_args.h"
#include "kernels/dispatch/low_latency/launch.hpp"
#include "kernels/combine/low_latency/combine_layout.h"
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "kernels/combine/low_latency/launch.hpp"

extern "C" void ll_bench_entry_gate_do(void* stream, uint8_t* flag);

namespace {
void check(int status, const char* operation)
{
    if (status) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
struct Buffer {
    uint8_t* p = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n)
    {
        check(aclrtMalloc(reinterpret_cast<void**>(&p), n, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~Buffer()
    {
        if (p) aclrtFree(p);
    }
    Buffer(const Buffer&) = delete;
    void upload(const void* data, size_t n)
    {
        if (n > bytes) throw std::runtime_error("upload overflow");
        check(aclrtMemcpy(p, bytes, data, n, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    }
    void download(void* data, size_t n)
    {
        if (n > bytes) throw std::runtime_error("download overflow");
        check(aclrtMemcpy(data, n, p, n, ACL_MEMCPY_DEVICE_TO_HOST), "download");
    }
};
struct Event {
    aclrtEvent value = nullptr;
    Event()
    {
        check(aclrtCreateEventWithFlag(&value, ACL_EVENT_TIME_LINE), "create timing event");
    }
    ~Event()
    {
        if (value) aclrtDestroyEvent(value);
    }
    void record(aclrtStream stream)
    {
        check(aclrtRecordEvent(value, stream), "record event");
    }
};
struct EntryGate {
    Buffer flag{64U};
    uint32_t* released = nullptr;
    aclrtStream control = nullptr;
    EntryGate()
    {
        check(aclrtMallocHost(reinterpret_cast<void**>(&released), 64U), "gate host allocation");
        std::memset(released, 0, 64U);
        released[0] = 1U;
        const auto status = aclrtCreateStream(&control);
        if (status != ACL_SUCCESS) {
            aclrtFreeHost(released);
            released = nullptr;
            check(status, "gate control stream");
        }
    }
    ~EntryGate()
    {
        if (control) aclrtDestroyStream(control);
        if (released) aclrtFreeHost(released);
    }
    void enqueue(aclrtStream stream)
    {
        // Previous iteration is complete. The release DMA uses a separate stream.
        check(aclrtMemset(flag.p, flag.bytes, 0, flag.bytes), "arm entry gate");
        ll_bench_entry_gate_do(stream, flag.p);
        check(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream), "enqueue device entry barrier");
    }
    void release()
    {
        check(aclrtMemcpyAsync(flag.p, flag.bytes, released, 64U, ACL_MEMCPY_HOST_TO_DEVICE, control),
              "release entry gate");
        check(aclrtSynchronizeStream(control), "gate release completion");
    }
};
uint8_t* local_base(const llbench::Runtime& session)
{
    return static_cast<uint8_t*>(session.workspace) + uint64_t(session.options.rank) * 1024U;
}
DispatchDedup::CommArgs comm_args(const llbench::Runtime& session)
{
    DispatchDedup::CommArgs args{};
    args.rank = session.options.rank;
    args.rankSize = session.options.world;
    args.localRankSize = DispatchDedup::LlGroupForWorld(session.options.world);
    args.localRank = session.options.rank % args.localRankSize;
    args.reserved0 = DispatchDedup::kLlPhysicalRanksPerHost;
    args.peerMemBytes = session.capacity;
    args.fftsVal = session.ffts;
    args.reservedAddresses[0] = static_cast<uint8_t*>(session.workspace);
    args.reservedAddresses[1] = static_cast<uint8_t*>(session.rdmaStage);
    args.reservedConfig[0] = LlRdma::kBackend;
    args.reservedConfig[1] = session.rdmaQps;
    for (int peer = 0; peer < session.options.world; ++peer) {
        if (!DispatchDedup::LlSameHost(session.options.rank, peer)) continue;
        auto base = static_cast<uint8_t*>(peer == session.options.rank ? session.workspace
                                                                       : aclshmem_ptr(session.workspace, peer));
        if (!base) throw std::runtime_error("Missing local MTE mapping");
        args.peerMems[peer] = base + uint64_t(peer) * 1024U;
    }
    return args;
}
struct Case {
    std::vector<std::unique_ptr<Buffer>> allocations;
    deepep::EpServerDedupKernelArgs dispatch{};
    deepep::EpServerDedupCombineKernelArgs combine{};
    std::vector<uint16_t> input;
    uint32_t variant = 0, bs = 0;
    Buffer* output = nullptr;
    Buffer& allocate(size_t bytes)
    {
        allocations.emplace_back(std::make_unique<Buffer>(bytes));
        return *allocations.back();
    }
    template <class T>
    uint8_t* upload(const T& data)
    {
        auto& b = allocate(sizeof(data));
        b.upload(&data, sizeof(data));
        return b.p;
    }
    Case(const llbench::Options& o, uint32_t b, uint32_t cores, uint8_t* comm) : bs(b)
    {
        DispatchDedup::EpMemoryServerDedupLayout layout{};
        CombineDedup::Layout combineLayout{};
        if (DispatchDedup::BuildDispatchServerDedupLayout(o.world, b, o.hidden, o.topk, o.experts, &layout,
                                                          DispatchDedup::EpServerDedupInputMode::Plain16, cores) != 0 ||
            !DispatchDedup::DispatchDedupPacketUbFits(layout, b, o.topk, o.experts, o.world, cores) ||
            !CombineDedup::BuildLayout(o.world, b, o.hidden, o.topk, o.experts, combineLayout, cores))
            throw std::runtime_error("Case exceeds kernel shape, UB or SHMEM limits");
        auto indices = llbench::routes(o, b);
        std::vector<float> weights(b * o.topk, 1.0f / o.topk);
        input.resize(b * o.hidden);
        for (uint32_t t = 0; t < b; ++t)
            for (uint32_t h = 0; h < o.hidden; ++h)
                input[t * o.hidden + h] = llbench::encode(llbench::input_value(o.rank, t, h), o.dtype == "bf16");
        auto& x = allocate(input.size() * 2);
        x.upload(input.data(), x.bytes);
        dispatch.x = x.p;
        auto& ids = allocate(indices.size() * 4);
        ids.upload(indices.data(), ids.bytes);
        dispatch.expertIds = ids.p;
        auto& w = allocate(weights.size() * 4);
        w.upload(weights.data(), w.bytes);
        dispatch.expertScales = w.p;
        dispatch.commArgs = comm;
        dispatch.coreNum = cores;
        dispatch.expandXOut = allocate(layout.expandXBytes).p;
        dispatch.expertTokenNumsOut = allocate(layout.expertTokenNumsBytes).p;
        dispatch.sendCountsOut = allocate(layout.sendCountsBytes).p;
        dispatch.tokenTypeOut = allocate(layout.tokenTypeBytes).p;
        dispatch.destinationIndexOut = allocate(layout.destinationIndexBytes).p;
        dispatch.relayReadIndexOut = allocate(layout.relayReadIndexBytes).p;
        dispatch.sourceMaskOut = allocate(layout.sourceMaskBytes).p;
        dispatch.expertScalesOut = allocate(layout.weightBytes).p;
        DispatchDedup::EpServerDedupTilingData d{};
        d.bs = b;
        d.h = o.hidden;
        d.topK = o.topk;
        d.moeExpertNum = o.experts;
        d.numMaxTokensPerRank = o.capacity;
        d.expertTokenNumsType = 1;
        d.dtype = o.dtype == "bf16" ? DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16
                                    : DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
        d.expandXOutDtype = d.dtype;
        d.inputMode = uint8_t(DispatchDedup::EpServerDedupInputMode::Plain16);
        d.tilingKey = DispatchDedup::LlTilingKeyForWorld(o.world);
        dispatch.tilingData = upload(d);
        variant = DispatchDedup::LlDispatchVariantForTiling(d.dtype, d.quantMode, d.expandXOutDtype, d.inputMode,
                                                            d.tilingKey);
        if (variant == DispatchDedup::kLlDispatchInvalidVariant)
            throw std::runtime_error("Unsupported Dispatch specialization");
        combine.commArgs = comm;
        combine.coreNum = cores;
        combine.expertOut = dispatch.expandXOut;
        combine.destinationIndex = dispatch.destinationIndexOut;
        combine.sendCounts = dispatch.sendCountsOut;
        combine.expertScales = dispatch.expertScalesOut;
        combine.tokenType = dispatch.tokenTypeOut;
        combine.relayReadIndex = dispatch.relayReadIndexOut;
        combine.sourceMask = dispatch.sourceMaskOut;
        output = &allocate(input.size() * 2);
        combine.yOut = output->p;
        CombineDedup::EpServerDedupCombineTilingData c{};
        c.bs = b;
        c.h = o.hidden;
        c.topK = o.topk;
        c.moeExpertNum = o.experts;
        c.dtype = d.dtype;
        c.numMaxTokensPerRank = o.capacity;
        c.tilingKey = d.tilingKey;
        combine.tilingData = upload(c);
    }
    double precision(bool bf16)
    {
        std::vector<uint16_t> actual(input.size());
        output->download(actual.data(), actual.size() * 2);
        double worst = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            double expected = llbench::decode(input[i], bf16), value = llbench::decode(actual[i], bf16);
            double error = std::abs(value - expected);
            worst = std::max(worst, error);
            // This roundtrip uses identity experts and uniform weights. Partial
            // results retain the operator's BF16/FP16 rounding at each stage.
            double tolerance = (bf16 ? .02 : .002) * (1 + std::abs(expected));
            if (!std::isfinite(value) || error > tolerance)
                throw std::runtime_error("Roundtrip precision failed at element " + std::to_string(i));
        }
        return worst;
    }
};
}  // namespace

LL_BENCH_EXPORT void ll_bench_run(const llbench::Runtime* shared)
{
    static_assert(llbench::workspaceBytes == DispatchDedup::kEpServerDedupPeerBytes);
    const auto& session = *shared;
    const auto& peers = *shared->peers;
    const auto& o = shared->options;
    const auto stream = shared->stream;
    const auto cores = shared->cores;
    if (DEEPEP_LL_COMBINE_ORIGINAL && o.world == 2)
        throw std::runtime_error("Unmodified upstream Combine supports G4/G16; use --variant opt for EP2 smoke tests");
    const auto recv = CombineDedup::EpCombineRecvCoreCount(cores);
    if (!recv || deepep_ll_combine_recv_cores() != DEEPEP_LL_COMBINE_RECV_CORES ||
        bool(deepep_ll_combine_early_reduce_enabled()) != CombineDedup::EpCombineEarlyReduce(1))
        throw std::runtime_error("Unsupported receive-core count or mismatched Host/device build");
    const bool profileBuild = deepep_ll_combine_profile_enabled() != 0;
    const uint32_t returnOptions = deepep_ll_combine_return_options();
    if (returnOptions != (DEEPEP_BENCH_REMOTE_FIRST | (DEEPEP_BENCH_DEFER_RETURN << 1)))
        throw std::runtime_error("Mismatched Host/device return options");
    if (o.profile && !profileBuild)
        throw std::runtime_error("Rebuild with build.sh --profile ON before requesting device diagnostics");
    std::ostringstream identity;
    identity << o.signature() << '/' << cores << '/' << DEEPEP_BENCH_SOURCE_ID << '/'
             << deepep_ll_combine_small_batch_enabled() << '/' << deepep_ll_combine_recv_cores() << '/'
             << deepep_ll_combine_early_reduce_enabled() << '/' << profileBuild << '/' << DEEPEP_LL_COMBINE_ORIGINAL
             << '/' << returnOptions;
    peers.verify(llbench::hash(identity.str()));
    check(aclrtSynchronizeStream(stream), "initial completion");
    peers.barrier();
    auto hostComm = comm_args(session);
    Buffer comm(sizeof(hostComm));
    comm.upload(&hostComm, sizeof(hostComm));
    std::unique_ptr<EntryGate> gate;
    if (o.entrySync == "device") {
        // Complete first-use SDK setup before blocking the compute stream.
        check(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream), "prime device entry barrier");
        check(aclrtSynchronizeStream(stream), "entry barrier preparation");
        gate = std::make_unique<EntryGate>();
    }
    std::filesystem::create_directories(o.output);
    auto rankPath = std::filesystem::path(o.output) / ("rank-" + std::to_string(o.rank) + ".csv");
    if (std::filesystem::exists(rankPath)) throw std::runtime_error("Result exists; choose a new output directory");
    std::ofstream samples(rankPath);
    if (!samples) throw std::runtime_error("Cannot open rank output");
    samples << "tokens,iteration,combine_event_us,original,entry_sync,profile_build,profile_level\n";
    samples << std::fixed << std::setprecision(3);
    std::ofstream profileCsv;
    if (o.profile) {
        profileCsv.open(std::filesystem::path(o.output) / ("rank-" + std::to_string(o.rank) + "-profile.csv"));
        if (!profileCsv) throw std::runtime_error("Cannot open profile output");
        llbench::profile_header(profileCsv);
    }
    std::ofstream summary;
    if (o.rank == 0) {
        summary.open(std::filesystem::path(o.output) / "summary.csv");
        if (!summary) throw std::runtime_error("Cannot open summary");
        summary << "ep,tokens,hidden,topk,experts,dtype,route,seed,capacity,warmup,source_id,samples,active_"
                   "aiv,recv_cores,small_batch,early_"
                   "reduce,remote_first,defer_return,original,entry_sync,profile_build,profile_level,rank_max_mean_us,"
                   "rank_max_min_us,"
                   "rank_max_max_us,p50_us,p99_us\n";
    }
    std::cout << "rank=" << o.rank << " EP=" << o.world << " AIV=" << cores << " receive=" << recv
              << " small_batch=" << deepep_ll_combine_small_batch_enabled()
              << " early_reduce=" << deepep_ll_combine_early_reduce_enabled()
              << " remote_first=" << (returnOptions & 1U) << " defer_return=" << ((returnOptions >> 1) & 1U)
              << " original=" << DEEPEP_LL_COMBINE_ORIGINAL << " entry_sync=" << o.entrySync
              << " profile_build=" << profileBuild << " profile_level=" << o.profile << std::endl;
    if (o.rank == 0 && profileBuild)
        std::cout << "Diagnostic build: use --profile OFF when rebuilding for clean Event performance." << std::endl;
    for (auto bs : o.tokens) {
        Case c(o, bs, cores, comm.p);
        peers.barrier();
        dispatch_server_dedup_kernel_do_v2(stream, &c.dispatch, c.variant);
        check(aclrtSynchronizeStream(stream), "Dispatch completion");
        peers.barrier();
        const uint64_t blocks = uint64_t(bs) * o.topk * ((uint64_t(o.hidden) * 2 + 479) / 480);
        combine_clear_result_flags_kernel_do(stream, local_base(session), blocks, cores);
        check(aclrtSynchronizeStream(stream), "Combine window preparation");
        peers.barrier();
        auto launch = [&]() { combine_server_dedup_kernel_do(stream, &c.combine); };
        launch();
        check(aclrtSynchronizeStream(stream), "correctness Combine");
        auto error = c.precision(o.dtype == "bf16");
        peers.barrier();
        for (uint32_t i = 0; i < o.warmup; ++i) {
            if (gate)
                gate->enqueue(stream);
            else
                peers.barrier();
            launch();
            if (gate) {
                peers.barrier();
                gate->release();
            }
            check(aclrtSynchronizeStream(stream), "warmup");
        }
        std::vector<float> times;
        times.reserve(o.iterations);
        std::unique_ptr<Buffer> profileBuffer;
        std::vector<uint64_t> profileData;
        if (o.profile) {
            profileData.resize(size_t(cores) * CombineProfile::kWords);
            profileBuffer = std::make_unique<Buffer>(profileData.size() * sizeof(uint64_t));
            c.combine.stageTimer = profileBuffer->p;
            c.combine.timerMode = o.profile;
        }
        Event start, end;
        for (uint32_t i = 0; i < o.iterations; ++i) {
            // Every peer completed its previous kernel before this barrier
            // releases either rotating bank. Host coordination is untimed.
            if (o.profile) {
                // Clear outside the Event so missing core exports cannot reuse an older sample.
                std::fill(profileData.begin(), profileData.end(), 0U);
                profileBuffer->upload(profileData.data(), profileBuffer->bytes);
            }
            if (gate)
                gate->enqueue(stream);
            else
                peers.barrier();
            start.record(stream);
            launch();
            end.record(stream);
            // Every rank has queued gate -> device barrier -> both Events
            // and Combine before Host releases any gate. MPI return skew
            // is absorbed by the device barrier outside the Event interval.
            if (gate) {
                peers.barrier();
                gate->release();
            }
            check(aclrtSynchronizeStream(stream), "sample completion");
            float ms = 0;
            check(aclrtEventElapsedTime(&ms, start.value, end.value), "event elapsed");
            if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid Event duration");
            times.push_back(ms * 1000);
            samples << bs << ',' << i << ',' << times.back() << ',' << DEEPEP_LL_COMBINE_ORIGINAL << ',' << o.entrySync
                    << ',' << profileBuild << ',' << o.profile << '\n';
            if (o.profile) {
                profileBuffer->download(profileData.data(), profileBuffer->bytes);
                llbench::profile_rows(profileCsv, profileData, cores, recv, o.rank, o.world, bs, o.device, o.profile,
                                      i);
            }
        }
        c.precision(o.dtype == "bf16");
        samples.flush();
        if (o.profile) profileCsv.flush();
        auto data = peers.gather(times.data(), times.size() * sizeof(float));
        if (o.rank == 0) {
            std::vector<float> rankMax(o.iterations, 0);
            for (uint32_t r = 0; r < o.world; ++r)
                for (uint32_t i = 0; i < o.iterations; ++i) {
                    float v;
                    std::memcpy(&v, data.data() + (r * o.iterations + i) * sizeof(float), sizeof(float));
                    rankMax[i] = std::max(rankMax[i], v);
                }
            auto s = llbench::stats(rankMax);
            summary << o.world << ',' << bs << ',' << o.hidden << ',' << o.topk << ',' << o.experts << ',' << o.dtype
                    << ',' << o.route << ',' << o.seed << ',' << o.capacity << ',' << o.warmup << ','
                    << DEEPEP_BENCH_SOURCE_ID << ',' << o.iterations << ',' << cores << ',' << recv << ','
                    << deepep_ll_combine_small_batch_enabled() << ',' << deepep_ll_combine_early_reduce_enabled() << ','
                    << (returnOptions & 1U) << ',' << ((returnOptions >> 1) & 1U) << ',' << DEEPEP_LL_COMBINE_ORIGINAL
                    << ',' << o.entrySync << ',' << profileBuild << ',' << o.profile << ',' << s.mean << ',' << s.min
                    << ',' << s.max << ',' << s.p50 << ',' << s.p99 << '\n';
            summary.flush();
            std::cout << "BS=" << bs << " rank-MAX Event us: mean=" << s.mean << " range=[" << s.min << ',' << s.max
                      << "] p99=" << s.p99 << " rank0_max_abs_error=" << error << std::endl;
        }
        peers.barrier();
    }
}
