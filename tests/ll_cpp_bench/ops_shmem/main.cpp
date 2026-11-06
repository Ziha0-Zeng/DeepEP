#include "fixture.hpp"
#include "../mpi_coordinator.hpp"
#include "../variant_api.hpp"
#include "runtime/ll_core_config.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

extern "C" void ll_bench_entry_gate_do(void* stream, uint8_t* flag);
namespace {
void check(int rc, const char* operation)
{
    if (rc) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}
struct Buffer {
    uint8_t* p = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(std::max(n, size_t(32)))
    {
        check(aclrtMalloc(reinterpret_cast<void**>(&p), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        check(aclrtMemset(p, bytes, 0, bytes), "clear allocation");
    }
    ~Buffer()
    {
        if (p) aclrtFree(p);
    }
    Buffer(const Buffer&) = delete;
    void upload(const void* data, size_t n)
    {
        if (n) check(aclrtMemcpy(p, bytes, data, n, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    }
};
struct Event {
    aclrtEvent value = nullptr;
    Event()
    {
        check(aclrtCreateEventWithFlag(&value, ACL_EVENT_TIME_LINE), "create Event");
    }
    ~Event()
    {
        if (value) aclrtDestroyEvent(value);
    }
    void record(aclrtStream stream)
    {
        check(aclrtRecordEvent(value, stream), "record Event");
    }
};
struct Gate {
    Buffer flag{64};
    uint32_t* released = nullptr;
    aclrtStream control = nullptr;
    Gate()
    {
        check(aclrtMallocHost(reinterpret_cast<void**>(&released), 64), "gate Host allocation");
        std::memset(released, 0, 64);
        released[0] = 1;
        check(aclrtCreateStream(&control), "gate control stream");
    }
    ~Gate()
    {
        if (control) aclrtDestroyStream(control);
        if (released) aclrtFreeHost(released);
    }
    void enqueue(aclrtStream stream)
    {
        check(aclrtMemset(flag.p, flag.bytes, 0, flag.bytes), "arm gate");
        ll_bench_entry_gate_do(stream, flag.p);
        check(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream), "device entry barrier");
    }
    void release()
    {
        check(aclrtMemcpyAsync(flag.p, flag.bytes, released, 64, ACL_MEMCPY_HOST_TO_DEVICE, control), "release gate");
        check(aclrtSynchronizeStream(control), "gate release completion");
    }
};
struct DeviceCase {
    opsbench::Fixture data;
    opsbench::Args args{};
    std::vector<std::unique_ptr<Buffer>> buffers;
    Buffer& allocate(size_t n)
    {
        buffers.emplace_back(std::make_unique<Buffer>(n));
        return *buffers.back();
    }
    template <class T>
    uint8_t* upload(const std::vector<T>& v)
    {
        auto& b = allocate(v.size() * sizeof(T));
        b.upload(v.data(), v.size() * sizeof(T));
        return b.p;
    }
    DeviceCase(const llbench::Options& o, uint32_t bs, uint32_t cores, uint64_t ub, uint8_t* context) : data(o, bs)
    {
        args.context = context;
        args.cores = cores;
        args.bf16 = o.dtype == "bf16";
        args.expanded = upload(data.expanded);
        args.experts = upload(data.experts);
        args.assist = upload(data.assist);
        args.counts = upload(data.counts);
        args.weights = upload(data.weights);
        args.expandedWeights = upload(data.expandedWeights);
        args.output = allocate(data.expected.size() * 2).p;
        opsbench::Config c{o.rank,    o.world,   bs,    o.hidden, o.topk,
                           o.experts, data.rows, cores, ub,       shmemx_get_ffts_config()};
        auto& config = allocate(sizeof(c));
        config.upload(&c, sizeof(c));
        args.config = config.p;
    }
    double precision(bool bf16)
    {
        std::vector<uint16_t> actual(data.expected.size());
        check(aclrtMemcpy(actual.data(), actual.size() * 2, args.output, actual.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST),
              "read output");
        double worst = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double expected = llbench::decode(data.expected[i], bf16), value = llbench::decode(actual[i], bf16);
            const double error = std::abs(value - expected);
            worst = std::max(worst, error);
            if (!std::isfinite(value) || error > (bf16 ? .02 : .002) * (1 + std::abs(expected)))
                throw std::runtime_error("ops-shmem-base precision failed at element " + std::to_string(i));
        }
        return worst;
    }
};
opsbench::Context context_for(const llbench::Runtime& session)
{
    opsbench::Context c{};
    c.rank = session.options.rank;
    c.ranksPerServer = std::min(session.options.world, 16U);
    const uint32_t server = c.rank / c.ranksPerServer, servers = session.options.world / c.ranksPerServer;
    for (uint32_t peer = server * c.ranksPerServer; peer < (server + 1) * c.ranksPerServer; ++peer) {
        void* address = peer == c.rank ? session.workspace : aclshmem_ptr(session.workspace, peer);
        if (!address) throw std::runtime_error("Missing SHMEM local mapping");
        c.localWindows[peer] = reinterpret_cast<uint64_t>(address);
    }
    for (uint32_t s = 0, channel = 0; s < servers; ++s) {
        // Local symmetric addresses are intentional: SHMEM translates the target.
        c.remoteWindows[s] = reinterpret_cast<uint64_t>(session.workspace) + opsbench::windowBytes;
        if (s != server) {
            c.peers[channel] = s * c.ranksPerServer + c.rank % c.ranksPerServer;
            channel += 2;  // Preserve the upstream context's two entries per server.
        }
    }
    c.remoteWindowBytes = opsbench::windowBytes;
    return c;
}
}  // namespace

LL_BENCH_EXPORT void ll_bench_run(const llbench::Runtime* shared)
{
    static_assert(llbench::workspaceBytes == opsbench::workspaceBytes);
    const auto& session = *shared;
    const auto& peers = *shared->peers;
    const auto& o = shared->options;
    const auto stream = shared->stream;
    const auto cores = shared->cores;
    if (o.profile) throw std::runtime_error("ops-shmem-base is a clean kernel; use --profile 0");
    if ((o.hidden != 3584 && o.hidden != 7168) || o.topk != 8 || o.experts != 384)
        throw std::runtime_error("Initial ops-shmem baseline supports H3584/H7168, TopK8, E384");
    uint64_t ub = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub < 128 * 1024) throw std::runtime_error("Cannot query sufficient UB capacity");
    peers.verify(llbench::hash(o.signature() + "/ops-shmem/" + DEEPEP_BENCH_SOURCE_ID + "/" + std::to_string(cores) +
                               "/" + std::to_string(ub)));
    std::filesystem::create_directories(o.output);
    const auto directory = std::filesystem::path(o.output);
    const auto rankPath = directory / ("rank-" + std::to_string(o.rank) + ".csv");
    if (std::filesystem::exists(rankPath)) throw std::runtime_error("Result exists; choose a fresh run-id");
    std::ofstream samples(rankPath), summary;
    if (!samples) throw std::runtime_error("Cannot open samples");
    samples << "tokens,iteration,combine_event_us,implementation,entry_sync\n";
    if (o.rank == 0) {
        summary.open(directory / "summary.csv");
        if (!summary) throw std::runtime_error("Cannot open summary");
        summary << "ep,tokens,hidden,topk,experts,dtype,route,seed,capacity,warmup,samples,active_aiv,implementation,"
                   "source_id,upstream_revision,entry_sync,rank_max_mean_us,rank_max_min_us,rank_max_max_us,p50_us,p99_"
                   "us\n";
    }
    auto c = context_for(session);
    Buffer context(sizeof(c));
    context.upload(&c, sizeof(c));
    std::unique_ptr<Gate> gate;
    if (o.entrySync == "device") {
        check(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream), "prime entry barrier");
        check(aclrtSynchronizeStream(stream), "entry barrier preparation");
        gate = std::make_unique<Gate>();
    }
    if (!o.rank)
        std::cout << "ops-shmem-base: upstream " << LL_OPS_REV
                  << "; SHMEM QP Put + batch-boundary quiet; entry_sync=" << o.entrySync << std::endl;
    for (auto bs : o.tokens) {
        DeviceCase item(o, bs, cores, ub, context.p);
        auto launch = [&] { ops_shmem_combine_do(stream, &item.args); };
        peers.barrier();
        launch();
        check(aclrtSynchronizeStream(stream), "correctness Combine");
        double error = item.precision(o.dtype == "bf16");
        peers.barrier();
        Event start, end;
        std::vector<float> times;
        for (uint32_t i = 0; i < o.warmup + o.iterations; ++i) {
            if (gate)
                gate->enqueue(stream);
            else
                peers.barrier();
            if (i >= o.warmup) start.record(stream);
            launch();
            if (i >= o.warmup) end.record(stream);
            if (gate) {
                peers.barrier();
                gate->release();
            }
            check(aclrtSynchronizeStream(stream), "Combine completion");
            if (i >= o.warmup) {
                float ms = 0;
                check(aclrtEventElapsedTime(&ms, start.value, end.value), "elapsed Event");
                if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid Event duration");
                times.push_back(ms * 1000);
                samples << bs << ',' << i - o.warmup << ',' << times.back() << ",ops-shmem-base," << o.entrySync
                        << '\n';
            }
        }
        item.precision(o.dtype == "bf16");
        samples.flush();
        auto gathered = peers.gather(times.data(), times.size() * sizeof(float));
        if (!o.rank) {
            std::vector<float> rankMax(o.iterations, 0);
            for (uint32_t rank = 0; rank < o.world; ++rank)
                for (uint32_t i = 0; i < o.iterations; ++i) {
                    float value;
                    std::memcpy(&value, gathered.data() + (rank * o.iterations + i) * sizeof(float), sizeof(float));
                    rankMax[i] = std::max(rankMax[i], value);
                }
            auto s = llbench::stats(rankMax);
            summary << o.world << ',' << bs << ',' << o.hidden << ',' << o.topk << ',' << o.experts << ',' << o.dtype
                    << ',' << o.route << ',' << o.seed << ',' << o.capacity << ',' << o.warmup << ',' << o.iterations
                    << ',' << cores << ",ops-shmem-base," << DEEPEP_BENCH_SOURCE_ID << ',' << LL_OPS_REV << ','
                    << o.entrySync << ',' << s.mean << ',' << s.min << ',' << s.max << ',' << s.p50 << ',' << s.p99
                    << '\n';
            summary.flush();
            std::cout << "BS=" << bs << " rank-MAX Event us: mean=" << s.mean << " range=[" << s.min << ',' << s.max
                      << "] p99=" << s.p99 << " rank0_max_abs_error=" << error << std::endl;
        }
        peers.barrier();
    }
}
