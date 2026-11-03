// Copyright (c) 2026, Lu Lu

#include "ops/low_latency.hpp"
#include <acl/acl.h>
#include <pybind11/stl.h>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/session.hpp"
#include "runtime/ll_core_config.hpp"
#include "../kernels/dispatch/low_latency/common/ll_rdma_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_stage_timer_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_reference_profile_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_expert_timer_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include "../kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "../kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "../kernels/combine/low_latency/combine_layout.h"
#include "../kernels/combine/low_latency/combine_profile_layout.h"
#include "../kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "../kernels/dispatch/low_latency/launch.hpp"
#include "../kernels/combine/low_latency/launch.hpp"

namespace py = pybind11;
namespace deepep {
namespace {
using Shape = std::vector<int64_t>;
constexpr uint64_t kRankBaseBiasBytes = 1024U;

void check_acl(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

class TimelineInterval {
    aclrtEvent start_ = nullptr;
    aclrtEvent end_ = nullptr;

public:
    TimelineInterval()
    {
        check_acl(aclrtCreateEventWithFlag(&start_, ACL_EVENT_TIME_LINE), "create start Event");
        try {
            check_acl(aclrtCreateEventWithFlag(&end_, ACL_EVENT_TIME_LINE), "create end Event");
        } catch (...) {
            aclrtDestroyEvent(start_);
            start_ = nullptr;
            throw;
        }
    }
    ~TimelineInterval()
    {
        if (end_) aclrtDestroyEvent(end_);
        if (start_) aclrtDestroyEvent(start_);
    }
    void begin(aclrtStream stream)
    {
        check_acl(aclrtRecordEvent(start_, stream), "record start Event");
    }
    void end(aclrtStream stream)
    {
        check_acl(aclrtRecordEvent(end_, stream), "record end Event");
    }
    float finish_us(aclrtStream stream)
    {
        check_acl(aclrtSynchronizeStream(stream), "synchronize benchmark stream");
        float ms = 0.0F;
        check_acl(aclrtEventElapsedTime(&ms, start_, end_), "read benchmark Event");
        return ms * 1000.0F;
    }
};

// Benchmark-only local submission gate, using the CANN Notify API rather than
// an AICore spin kernel. HCCL supplies the cross-rank synchronization.
class BenchmarkLaunchGate {
    aclrtNotify notify_ = nullptr;
    aclrtStream control_ = nullptr;
    aclrtStream target_ = nullptr;
    bool armed_ = false;
    bool released_ = false;
public:
    BenchmarkLaunchGate()
    {
        check_acl(aclrtCreateNotify(&notify_, 0), "create benchmark launch notify");
        try {
            check_acl(aclrtCreateStream(&control_), "create benchmark release stream");
        } catch (...) {
            aclrtDestroyNotify(notify_);
            throw;
        }
    }
    ~BenchmarkLaunchGate()
    {
        // Also unblock a partially submitted stream if a gate callback throws.
        if (armed_ && !released_) aclrtRecordNotify(notify_, control_);
        if (armed_) aclrtSynchronizeStream(target_);
        if (control_) {
            aclrtSynchronizeStream(control_);
            aclrtDestroyStream(control_);
        }
        if (notify_) aclrtDestroyNotify(notify_);
    }
    void arm(aclrtStream stream)
    {
        target_ = stream;
        check_acl(aclrtWaitAndResetNotify(notify_, stream, 120000), "arm benchmark launch notify");
        armed_ = true;
    }
    void release()
    {
        check_acl(aclrtRecordNotify(notify_, control_), "release benchmark launch notify");
        released_ = true;
    }
};

uint32_t pin_session_cores(Session& session)
{
    session.workspace();
    const auto config = query_ll_core_config();
    if (config.device < 0) throw std::runtime_error("LL launch requires a current device for core verification");
    return session.pin_low_latency_cores(config.active);
}

uint8_t* peer_buffer_base(Session& session, int peer)
{
    if (peer < 0 || peer >= session.world()) throw std::invalid_argument("Invalid LL peer");
    if (session.ll_rdma() && !DispatchDedup::LlSameHost(
            session.rank(), peer, session.ll_ranks_per_host())) return nullptr;
    auto* original = static_cast<uint8_t*>(peer == session.rank() ? session.workspace() : aclshmem_ptr(session.workspace(), peer));
    if (original == nullptr) throw std::invalid_argument("Invalid LL peer mapping");
    if (session.capacity() != (8ULL << 30)) return original;
    // Translate the original symmetric pointer before applying the destination-rank
    // bias. The shared peer table keeps local receive and remote send views equal.
    return original + uint64_t(peer) * kRankBaseBiasBytes;
}
uint8_t* tensor(py::handle value, const Shape& shape, const char* dtype)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(value).cast<bool>()) {
        throw std::invalid_argument("LL requires tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(value);
    if (!obj.attr("is_contiguous")().cast<bool>() || obj.attr("shape").cast<Shape>() != shape ||
        !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("LL tensor shape, dtype or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
uint8_t* stage_timer(py::dict tensors, uint32_t cores, uint32_t& mode, bool allow_reference = false)
{
    mode = 1U;
    if (tensors.contains("ll_combine_profile")) {
        if (allow_reference || !deepep_ll_combine_profile_enabled())
            throw std::invalid_argument("Combine profiling requires a Combine call and DEEPEP_LL_COMBINE_PROFILE=ON");
        for (const auto* key : {"ll_stage_timer", "ll_reference_profile", "ll_expert_timer"})
            if (tensors.contains(key) && !tensors[key].is_none())
                throw std::invalid_argument("Combine profiling cannot be mixed with other tracing");
        mode = tensors["ll_combine_profile_level"].cast<uint32_t>();
        if (mode != 1U && mode != 2U) throw std::invalid_argument("Invalid Combine profile level");
        auto profile = py::reinterpret_borrow<py::object>(tensors["ll_combine_profile"]);
        auto device = profile.attr("device");
        if (device.attr("type").cast<std::string>() != "npu" ||
            !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
            throw std::invalid_argument("Combine profile must be on the input NPU");
        auto* pointer = tensor(profile, {cores, CombineProfile::kWords}, "int64");
        if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
            throw std::invalid_argument("Combine profile requires 512-byte alignment");
        return pointer;
    }
    if (tensors.contains("ll_reference_profile") && !tensors["ll_reference_profile"].is_none()) {
        if (!allow_reference) throw std::invalid_argument("LL reference profiling supports dispatch only");
        if ((tensors.contains("ll_stage_timer") && !tensors["ll_stage_timer"].is_none()) ||
            (tensors.contains("ll_expert_timer") && !tensors["ll_expert_timer"].is_none()))
            throw std::invalid_argument("LL reference profiling cannot be combined with stage/expert tracing");
        if (!deepep_ll_reference_profile_enabled() || deepep_ll_stage_timer_enabled())
            throw std::invalid_argument("Rebuild with DEEPEP_LL_REFERENCE_PROFILE=ON and DEEPEP_LL_STAGE_TIMER=OFF");
        auto profile = py::reinterpret_borrow<py::object>(tensors["ll_reference_profile"]);
        auto device = profile.attr("device");
        if (device.attr("type").cast<std::string>() != "npu" ||
            !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
            throw std::invalid_argument("LL reference profile must be on the input NPU");
        auto* pointer = tensor(profile, {cores, LlReferenceProfile::kSlots, LlReferenceProfile::kFields}, "int64");
        if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
            throw std::invalid_argument("LL reference profile requires 512-byte alignment");
        mode = 3U;  // Private launch selector; the reference record's boundary mode remains 1.
        return pointer;
    }
    if (!tensors.contains("ll_stage_timer") || tensors["ll_stage_timer"].is_none()) return nullptr;
    if (!deepep_ll_stage_timer_enabled()) throw std::invalid_argument("Legacy LL stage tracing was removed; use LL reference profiling");
    if (tensors.contains("ll_stage_timer_mode")) mode = tensors["ll_stage_timer_mode"].cast<uint32_t>();
    if (mode != 1U && mode != 2U) throw std::invalid_argument("Invalid LL timer mode");
    auto timer = py::reinterpret_borrow<py::object>(tensors["ll_stage_timer"]);
    auto device = timer.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("LL timer must be on the input NPU");
    auto* pointer = tensor(timer, {cores, LlStageTimer::kSlots, LlStageTimer::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
        throw std::invalid_argument("LL timer requires 512-byte alignment for independent core writeback");
    return pointer;
}
uint8_t* expert_timer(py::dict tensors, uint32_t cores, uint32_t experts, uint8_t* stage)
{
    if (!tensors.contains("ll_expert_timer") || tensors["ll_expert_timer"].is_none()) return nullptr;
    if (!stage || !deepep_ll_stage_timer_enabled())
        throw std::invalid_argument("Expert diagnostics require stage capture and timer ON");
    auto value=py::reinterpret_borrow<py::object>(tensors["ll_expert_timer"]);
    auto device=value.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("Expert timer must be on the input NPU");
    auto* pointer=tensor(value, {cores, LlExpertTimer::kHeaderRows + experts*LlExpertTimer::StageCount,
                                LlExpertTimer::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer)%512U)
        throw std::invalid_argument("Expert timer requires 512-byte alignment");
    return pointer;
}
DispatchDedup::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    const uint32_t coreNum = query_ll_core_config().active;
    DispatchDedup::EpMemoryServerDedupLayout result{};
    auto mode = fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                    : DispatchDedup::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 65536 ||
        DispatchDedup::BuildDispatchServerDedupLayout(p, b, h, k, e, &result, mode, coreNum) !=
            DispatchDedup::DISPATCH_DEDUP_SUCCESS ||
        !DispatchDedup::DispatchDedupPacketUbFits(result, b, k, e, p, coreNum) || result.capacityRows > INT32_MAX) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    return result;
}
template <class T>
py::bytes bytes(const T& data)
{
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}
uint8_t dtype_code(const std::string& dtype)
{
    if (dtype == "bfloat16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (dtype == "float16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (dtype == "float8_e4m3fn") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    throw std::invalid_argument("LL dispatch supports BF16/FP16 or packed FP8 E4M3FN");
}
static_assert(DispatchDedup::kLlDispatchFp16 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16 &&
              DispatchDedup::kLlDispatchBf16 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16 &&
              DispatchDedup::kLlDispatchE4M3 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 &&
              DispatchDedup::kLlDispatchE5M2 == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E5M2,
              "LL dispatch variant dtype values must match tiling");
static_assert(DispatchDedup::kLlDispatchPlain16 == uint8_t(DispatchDedup::EpServerDedupInputMode::Plain16) &&
              DispatchDedup::kLlDispatchQuantize16 == uint8_t(DispatchDedup::EpServerDedupInputMode::Quantize16) &&
              DispatchDedup::kLlDispatchPrequantized == uint8_t(DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs),
              "LL dispatch variant input mode values must match tiling");

// Both device tiling construction and Host launch selection use these fields.
// Selection needs no device readback and does not synchronize the stream.
DispatchDedup::EpServerDedupTilingData dispatch_format_tiling(int world, const std::string& dtype)
{
    DispatchDedup::EpServerDedupTilingData data{};
    const bool fp8 = dtype == "float8_e4m3fn";
    data.tilingKey = DispatchDedup::LlTilingKeyForWorld(world);
    data.dtype = dtype_code(dtype);
    data.quantMode = fp8 ? 4 : 0;
    data.inputMode = static_cast<uint8_t>(fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                                            : DispatchDedup::EpServerDedupInputMode::Plain16);
    data.expandXOutDtype = data.dtype;
    return data;
}
}  // namespace

void bind_low_latency(py::module_& module)
{
    if (deepep_ll_core_protocol_version() != DispatchDedup::kLlCoreProtocolVersion)
        throw std::runtime_error("LL Host/device core protocol mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_core_protocol") = DispatchDedup::kLlCoreProtocolVersion;
    module.attr("low_latency_roce_qps_per_peer") = 0; // actual active physical cores
    module.attr("low_latency_implementation") = "ndr-hns1825";
    module.attr("low_latency_combine_small_batch") = bool(deepep_ll_combine_small_batch_enabled());
    module.attr("low_latency_combine_profile_enabled") = bool(deepep_ll_combine_profile_enabled());
    module.def("low_latency_combine_profile_layout", []() {
        py::dict result;
        result["schema"] = CombineProfile::kSchema;
        result["words"] = CombineProfile::kWords;
        result["count"] = uint32_t(CombineProfile::Count);
        py::dict indices;
#define PROFILE_WORD(name) indices[#name] = uint32_t(CombineProfile::name)
        PROFILE_WORD(Entry); PROFILE_WORD(ProfileInitEnd); PROFILE_WORD(MagicEnd);
        PROFILE_WORD(InitCountEnd); PROFILE_WORD(InitLayoutEnd); PROFILE_WORD(InitEnd);
        PROFILE_WORD(StageBegin); PROFILE_WORD(StageSelected); PROFILE_WORD(StageCopiesEnd);
        PROFILE_WORD(StageDrained); PROFILE_WORD(StageEnd);
        PROFILE_WORD(NotifyBegin); PROFILE_WORD(NotifyLocalReady); PROFILE_WORD(NotifyEnd);
        PROFILE_WORD(ReadyBegin); PROFILE_WORD(ReadyEnd);
        PROFILE_WORD(PartialBegin); PROFILE_WORD(PartialInitEnd); PROFILE_WORD(PlanBegin);
        PROFILE_WORD(PlanEnd); PROFILE_WORD(PartialRunBegin); PROFILE_WORD(PartialRunEnd); PROFILE_WORD(PartialEnd);
        PROFILE_WORD(FinalBegin); PROFILE_WORD(FinalInitEnd); PROFILE_WORD(FinalFirstReady);
        PROFILE_WORD(FinalLastReady); PROFILE_WORD(FinalLastStore); PROFILE_WORD(FinalEnd); PROFILE_WORD(DrainEnd);
        PROFILE_WORD(DirectBegin); PROFILE_WORD(DirectPrepared); PROFILE_WORD(DirectFirstSubmit);
        PROFILE_WORD(DirectLastSubmit); PROFILE_WORD(DirectEnd); PROFILE_WORD(PipeDone); PROFILE_WORD(FlushDone);
        PROFILE_WORD(StageRows); PROFILE_WORD(DirectRows); PROFILE_WORD(PartialRows); PROFILE_WORD(FinalTokens);
        PROFILE_WORD(PartialScanTicks); PROFILE_WORD(PartialMetadataTicks); PROFILE_WORD(PartialGetWaitTicks);
        PROFILE_WORD(PartialComputeIssueTicks); PROFILE_WORD(PartialSubmitTicks);
        PROFILE_WORD(ProbeTicks); PROFILE_WORD(ProbeMissTicks); PROFILE_WORD(ProbeCalls); PROFILE_WORD(ProbeMisses);
        PROFILE_WORD(FinalComputeIssueTicks); PROFILE_WORD(RdmaPutTicks); PROFILE_WORD(RdmaQuietTicks);
        PROFILE_WORD(RdmaPackets); PROFILE_WORD(RdmaQuietCalls); PROFILE_WORD(Contributions);
        PROFILE_WORD(Rank); PROFILE_WORD(Core); PROFILE_WORD(Tokens); PROFILE_WORD(World);
        PROFILE_WORD(Schema); PROFILE_WORD(Complete); PROFILE_WORD(Valid); PROFILE_WORD(Level);
#undef PROFILE_WORD
        result["indices"] = indices;
        return result;
    });
    module.attr("low_latency_combine_recv_cores") = deepep_ll_combine_recv_cores();
    module.attr("low_latency_combine_early_reduce") = bool(deepep_ll_combine_early_reduce_enabled());
    module.attr("low_latency_combine_return_options") = deepep_ll_combine_return_options();
#ifdef DEEPEP_LL_RDMA
    if (!deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = true;
#else
    if (deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = false;
#endif
    module.attr("low_latency_stage_timer_enabled") = bool(deepep_ll_stage_timer_enabled());
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    constexpr uint32_t reference_enabled = 1U;
#else
    constexpr uint32_t reference_enabled = 0U;
#endif
    if (deepep_ll_reference_profile_enabled() != reference_enabled ||
        deepep_ll_reference_profile_schema() != LlReferenceProfile::kSchema ||
        (reference_enabled && deepep_ll_stage_timer_enabled()))
        throw std::runtime_error("LL Host/device reference profile mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_reference_profile_enabled") = bool(reference_enabled);
    module.def("low_latency_reference_profile_layout", []() {
        py::dict result;
        result["schema"] = LlReferenceProfile::kSchema;
        result["slots"] = LlReferenceProfile::kSlots;
        result["fields"] = LlReferenceProfile::kFields;
        std::vector<std::string> names;
        for (const auto* name : LlReferenceProfile::kNames) names.emplace_back(name);
        result["words"] = names;
        return result;
    });
    module.def("low_latency_expert_timer_layout", []() {
        py::dict result;
        result["schema"]=LlExpertTimer::kSchema;
        result["header_rows"]=LlExpertTimer::kHeaderRows;
        result["fields"]=LlExpertTimer::kFields;
        std::vector<std::string> names;
        for (const auto* name : LlExpertTimer::kNames) names.emplace_back(name);
        result["stages"]=names;
        return result;
    });
    module.def("low_latency_stage_timer_layout", []() {
        py::dict result;
        result["schema"] = LlStageTimer::kSchema;
        result["slots"] = LlStageTimer::kSlots;
        result["fields"] = LlStageTimer::kFields;
        std::vector<std::string> names;
        for (const auto* name : LlStageTimer::kNames) names.emplace_back(name);
        result["stages"] = names;
        return result;
    });
    module.attr("low_latency_dispatch_metadata_cores") = DispatchDedup::kLlMetadataCores;
    module.def("low_latency_core_config", []() {
        const auto cfg = query_ll_core_config();
        py::dict result;
        result["platform_aiv"] = cfg.platform; result["available_aiv"] = cfg.available;
        result["active_aiv"] = cfg.active; result["send_cores"] = cfg.active / 2U;
        result["recv_cores"] = cfg.active / 2U; result["device"] = cfg.device;
        result["runtime_device_queried"] = cfg.device >= 0;
        result["control_capacity_aiv"] = DispatchDedup::kLlMaxCoreNum;
        result["dispatch_metadata_cores"] = DispatchDedup::kLlMetadataCores;
        result["dispatch_split_valid"] = DispatchDedup::MetadataSplitValid(cfg.active);
        result["dispatch_send_cores"] = DispatchDedup::MetadataSplitValid(cfg.active) ? DispatchDedup::PayloadCores(cfg.active) : 0U;
        result["dispatch_recv_cores"] = DispatchDedup::MetadataSplitValid(cfg.active) ? DispatchDedup::PayloadCores(cfg.active) : 0U;
        return result;
    });
    for (uint32_t world : {8U, 16U, 32U, 64U, 128U, 256U})
        if (deepep_ll_ranks_per_group(world) != DispatchDedup::LlGroupForWorld(world))
            throw std::runtime_error("LL Host/device tiling-key policy mismatch; rebuild the complete wheel");
    module.def("low_latency_ranks_per_group", [](uint32_t world) { return DispatchDedup::LlGroupForWorld(world); });
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        dtype_code(dtype);
        try {
            auto plan = layout(p, b, h, k, e, dtype == "float8_e4m3fn");
            py::dict result;
            result["capacity_rows"] = plan.capacityRows;
            result["scale_packs"] = (h + 127) / 128;
            return py::object(result);
        } catch (const std::invalid_argument&) {
            return py::object(py::none());
        }
    });
    module.def("low_latency_comm_args", [](Session& session) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            session.world() > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE ||
            session.world() < 8 || session.world() % DispatchDedup::LlGroupForWorld(session.world()) != 0) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        pin_session_cores(session);
        DispatchDedup::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % DispatchDedup::LlGroupForWorld(session.world());
        args.localRankSize = DispatchDedup::LlGroupForWorld(session.world());
        args.reserved0 = session.ll_ranks_per_host();
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        if (session.ll_rdma()) {
            args.reservedAddresses[0] = static_cast<uint8_t*>(session.workspace());
            args.reservedAddresses[1] = static_cast<uint8_t*>(session.ll_rdma_stage());
            args.reservedConfig[0] = LlRdma::kBackend;
            args.reservedConfig[1] = session.ll_rdma_qps();
        }
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = peer_buffer_base(session, peer);
            if (session.ll_rdma() && !DispatchDedup::LlSameHost(
                    session.rank(), peer, session.ll_ranks_per_host())) continue;
            if (args.peerMems[peer] == nullptr || reinterpret_cast<uintptr_t>(args.peerMems[peer]) % 32) {
                throw std::invalid_argument("LL requires aligned MTE mappings for every peer");
            }
        }
        return bytes(args);
    });
    module.def("low_latency_dispatch_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype,
                                                 int num_max_tokens_per_rank) {
        const bool fp8 = dtype == "float8_e4m3fn";
        layout(p, b, h, k, e, fp8);
        auto data = dispatch_format_tiling(p, dtype);
        data.bs = b;
        data.h = h;
        data.numMaxTokensPerRank = num_max_tokens_per_rank;
        data.topK = k;
        data.moeExpertNum = e;
        data.expertTokenNumsType = 1;
        return bytes(data);
    });
    module.def("low_latency_combine_tiling",
               [](int p, int b, int h, int k, int e, const std::string& dtype, int num_max_tokens_per_rank) {
                   CombineDedup::Layout plan{};
                   if ((dtype != "bfloat16" && dtype != "float16") || !CombineDedup::BuildLayout(p, b, h, k, e, plan, query_ll_core_config().active)) {
                       throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
                   }
                   CombineDedup::EpServerDedupCombineTilingData data{};
                   data.tilingKey = DispatchDedup::LlTilingKeyForWorld(p);
        data.bs = b;
                   data.h = h;
                   data.numMaxTokensPerRank = num_max_tokens_per_rank;
                   data.topK = k;
                   data.moeExpertNum = e;
                   data.dtype = dtype_code(dtype);
                   return bytes(data);
               });
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto plan = layout(session.world(), b, h, k, e, fp8);
        const auto format = dispatch_format_tiling(session.world(), dtype);
        const uint32_t variant = DispatchDedup::LlDispatchVariantForTiling(
            format.dtype, format.quantMode, format.expandXOutDtype, format.inputMode, format.tilingKey);
        if (variant == DispatchDedup::kLlDispatchInvalidVariant)
            throw std::invalid_argument("Unsupported LL dispatch specialization");
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupKernelArgs args{};
        args.coreNum = pin_session_cores(session);
        args.stageTimer = stage_timer(tensors, args.coreNum, args.timerMode, true);
        args.expertTimer = expert_timer(tensors, args.coreNum, e, args.stageTimer);
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        args.x = get("input", {b, h}, dtype.c_str());
        args.expertIds = get("indices", {b, k}, "int32");
        args.expertScales = get("weights", {b, k}, "float32");
        args.expandXOut = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        args.dynamicScalesOut = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        args.expertTokenNumsOut = get("expert_recv_counts", {e / session.world()}, "int64");
        args.sendCountsOut = get("send_counts", {e}, "int32");
        args.tokenTypeOut = get("token_type", {c}, "int32");
        args.destinationIndexOut = get("destination_index", {c, 3}, "int32");
        args.relayReadIndexOut = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMaskOut = get("source_mask", {b}, "int32");
        args.expertScalesOut = get("source_weights", {b, k}, "float32");
        args.tilingData = get("tiling", {sizeof(DispatchDedup::EpServerDedupTilingData)}, "uint8");
        args.inputScales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        session.workspace();  // Validate the owning context before launch.
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        dispatch_server_dedup_kernel_do_v2(reinterpret_cast<void*>(stream_address), &args, variant);
    });
    module.def("clear_low_latency_combine_windows", [](Session& session, int b, int h, int k, int e,
                                                       uintptr_t stream_address) {
        CombineDedup::Layout plan{};
        if (!session.ll_rdma() || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            !CombineDedup::BuildLayout(session.world(), b, h, k, e, plan, query_ll_core_config().active))
            throw std::invalid_argument("Invalid NDR combine window preparation");
        const uint64_t blocks = uint64_t(b) * k * ((uint64_t(h) * 2U + 479U) / 480U);
        auto* local = peer_buffer_base(session, session.rank());
        const uint32_t cores = pin_session_cores(session);
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        py::gil_scoped_release release;
        // Called after local device completion and BEFORE HCCL completion
        // exchange. Clear both banks because the next per-core magic is device-owned.
        combine_clear_result_flags_kernel_do(stream, local, blocks, cores);
        const int status = aclrtSynchronizeStream(stream);
        if (status != 0) throw std::runtime_error("NDR combine result initialization failed: " + std::to_string(status));
    });
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype, uintptr_t stream_address,
                                         bool benchmark_device_barrier,
                                         py::object benchmark_gate) -> py::object {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16") ||
            !CombineDedup::BuildLayout(session.world(), b, h, k, e, plan, query_ll_core_config().active)) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        const int64_t c = plan.capacityRows;
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupCombineKernelArgs args{};
        args.coreNum = pin_session_cores(session);
        args.stageTimer = stage_timer(tensors, args.coreNum, args.timerMode);
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.expertScales = get("source_weights", {b, k}, "float32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {b}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.workspace();
        float elapsed_us = 0.0F;
        session.invalidate_dispatch_controls();
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        if (benchmark_device_barrier) {
            TimelineInterval timer;
            BenchmarkLaunchGate launch_gate;
            launch_gate.arm(stream);
            // Invoke the public HCCL gate only AFTER all Python tensor validation.
            // Its Work.wait() adds a dependency to the current runtime stream.
            // Host coordination, gate submission and completion are outside Events.
            if (!benchmark_gate.is_none()) {
                benchmark_gate();
            } else {
                if (session.ll_rdma() && session.world() > int(session.ll_ranks_per_host()))
                    throw std::invalid_argument("Cross-host timing requires an HCCL device gate");
                check_acl(aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream),
                          "enqueue benchmark device barrier");
            }
            py::gil_scoped_release release;
            timer.begin(stream);
            combine_server_dedup_kernel_do(stream, &args);
            timer.end(stream);
            // Everything is queued before any local task can execute. Each rank
            // releases its own Notify, then HCCL absorbs inter-rank release skew.
            launch_gate.release();
            elapsed_us = timer.finish_us(stream);
        } else {
            py::gil_scoped_release release;
            combine_server_dedup_kernel_do(stream, &args);
        }
        return benchmark_device_barrier ? py::cast(elapsed_us) : py::none();
    });
}
}  // namespace deepep
