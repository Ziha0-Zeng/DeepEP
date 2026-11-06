// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H
#define COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H

#include "kernel_operator.h"
#include "combine_profile_device.h"
#include "../../dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
#include "../../dispatch/low_latency/common/comm_args.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "ep_memory_server_dedup_combine_config.h"
#include "combine_server_dedup_partial.h"
#include "combine_stage_wire.h"
#include "combine_server_dedup_source_aggregate.h"
#include "combine_server_dedup_select.h"

namespace CombineDedup {
using namespace AscendC;

// Local stage, readiness, DIRECT/partial return and source-token aggregation.
template<uint32_t Group>
class CombineMemoryServerDedupKernel {
    friend struct CombineModuleTestAccess;

public:
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* profile)
    {
        profile_.Bind(profile);
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profileOwner_ = profile;
        returnBatch_.SetCombineProfile(profile);
#endif
    }
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR sendCounts,
                                GM_ADDR expertScales, GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask,
                                GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t topK, uint32_t experts, uint8_t dtype,
                                uint8_t quantMode, int64_t magic, bool enableDedup = true)
    {
        (void)quantMode;

        core_ = GetBlockIdx();
        coreNum_ = GetBlockNum();
        recvCores_ = EpCombineRecvCoreCount(coreNum_);
        if (recvCores_ == 0U) { Trap(); return; }
        sendCores_ = coreNum_ - recvCores_;
        notifyBegin_ = coreNum_ - DispatchDedup::kLlNotifyCoreNum;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        enableDedup_ = enableDedup;
        h_ = h;
        topK_ = topK;
        bs_ = bs;
        experts_ = experts;
        dtype_ = dtype;
        expertAddr_ = expertOut;
        tableAddr_ = assistInfoForCombine;
        relayAddr_ = relayReadIndex;
        typesAddr_ = tokenType;
        weightsAddr_ = expertScales;
        masksAddr_ = sourceMask;
        outputAddr_ = yOut;
        // Load the device-produced prefix through MTE2 rather than caching a
        // mutable GM scalar across launches. This 32B temporary becomes the
        // independent M2 prepare sum only after the scalar count is consumed.
        pipe_.InitBuffer(arena_, core_ >= sendCores_ ? kEpServerDedupCombineRecvUbBytes
                                                                           : kEpServerDedupCombineUbBytes);
        prepareSum_ = Allocate<float>(32U);
        GlobalTensor<int32_t> prefix;
        prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCounts));
        auto tokenCount = prepareSum_.ReinterpretCast<int32_t>();
        const uint64_t prefixStart = profile_.DetailClock();
        DataCopyPad(tokenCount, prefix[experts - 1U], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        Sync<HardEvent::MTE2_S>();
        profile_.Elapsed(CombineProfile::InitPrefixTicks, prefixStart);
        profile_.Mark(CombineProfile::InitCountEnd);
        const int32_t tokens = tokenCount.GetValue(0U);
        const uint32_t localExperts = experts / uint32_t(args_->rankSize);
        const uint32_t copies = topK < localExperts ? topK : localExperts;
        const uint64_t capacity = uint64_t(args_->rankSize) * bs * copies;
        if (tokens < 0 || uint32_t(tokens) > kEpServerDedupCombineMaxTokenNum || uint64_t(tokens) > capacity) {
            Trap();  // Invalid device data must not become a truncated successful launch.
            return;
        }
        tokens_ = uint32_t(tokens);
        lanes_ = EpCombinePayloadLaneCount(tokens_, h_);
        workspace_ = EpCombineM45Layout(tokens_, bs_, h_, topK_, recvCores_);
        if (lanes_ == 0U || workspace_.inputLanes == 0U) {
            Trap();
            return;
        }

        profile_.Mark(CombineProfile::InitLayoutEnd);
        blocks_ = (h_ * 2U + 479U) / 480U;
        laneBytes_ = blocks_ * 480U;
        resultStride_ = blocks_ * 512U;
        window_ = (uint64_t(magic) % 2U) * DispatchDedup::kEpServerDedupWindowBytes;
        control_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET;
        stageStride_ = resultStride_;
        stageBase_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + uint64_t(bs) * topK * resultStride_;
        expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(expertOut));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(assistInfoForCombine));
        types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenType));
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(args_->peerMems[args_->rank] + stageBase_));
        if (LlTransport::Rdma(args_)) stage_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        preparePublish_ = Allocate<float>(256U);
        prepareInput_ = Allocate<float>(kCombinePrepareRowBytes + 256U);
        prepareWork_ = Allocate<float>(512U);
        prepareZero_ = Allocate<float>(kCombinePrepareRowBytes);
        readyInput_ = Allocate<float>(kCombineReadyInputBytes);
        readyWork_ = Allocate<float>(256U);
        readySum_ = Allocate<float>(32U);
        readyZero_ = Allocate<float>(512U);
        notifyUb_ = Allocate<float>(kCombineNotifyRows * 32U);
        const uint32_t rowBytes = EpCombineAlign(tokens_ == 0U ? 4U : tokens_ * 4U, 256U);
        typeUb_ = Allocate<int32_t>(rowBytes);
        indexUb_ = Allocate<int32_t>(rowBytes);
        packedUb_ = Allocate<int32_t>(rowBytes);
        maskUb_ = Allocate<uint8_t>(EpCombineAlign((tokens_ + 7U) / 8U + 32U, 256U));
        tableUb_ = Allocate<int32_t>(kEpServerDedupCombineTableRows * 12U);
        payloadUb_ = Allocate<uint16_t>(lanes_ * laneBytes_);
        resultUb_ = Allocate<uint8_t>(lanes_ * resultStride_);
    }

    // 完整combine调度：本地待归约行准备后，发送核处理DIRECT回传，接收核处理ready与归约。
    // 去重路径执行通知→等待→partial回传(M4)→源token加权归约(M5)；非去重走直接聚合(M6)。
    // 下列阶段属于同一个设备kernel，复用UB和完成事件，不能当作独立launch相减计时。
    __aicore__ inline void Process()
    {
        if (enableDedup_) {

            profile_.Mark(CombineProfile::StageBegin);
            CopyLocalPendingTokens();
            profile_.Mark(CombineProfile::StageEnd);

        }
        if (core_ >= sendCores_) {
            if (enableDedup_) {
                if (core_ >= notifyBegin_) {

                    NotifyServerCard();

                    // Notification has been issued; do not add an MTE3 drain.
                }

#ifndef DEEPEP_LL_COMBINE_SMALL_BATCH
                WaitServerReady();
#endif

                ReturnPartialTokens();

                profile_.Mark(CombineProfile::FinalBegin);
                const uint32_t retiredOutputs = AggregateSourceTokens();
                profile_.Mark(CombineProfile::FinalEnd);

                DrainSharedOutputs(retiredOutputs);
                profile_.Mark(CombineProfile::DrainEnd);

            } else {

                profile_.Mark(CombineProfile::FinalBegin);
                AggregateSourceTokensM6();

            }
        } else {

            profile_.Mark(CombineProfile::DirectBegin);
            ReturnDirectTokens();
            profile_.Mark(CombineProfile::DirectEnd);

        }
        // Retires prepare publication, both zero sources and server notification;
        // no arena/pipe reset or exit may precede completion of their DMA reads.

        PipeBarrier<PIPE_ALL>();
        // All resident control sources and output lanes have retired. Pending
        // return GM slots are independent of M5 UB and are reclaimed only here.
        profile_.Mark(CombineProfile::PipeDone);
        returnBatch_.Flush(arena_.Get<uint8_t>());
        profile_.Mark(CombineProfile::FlushDone);
    }

private:
    __aicore__ inline void AggregateSourceTokensM6()
    {
        ServerDedupSourceAggregate aggregate;
#ifdef DEEPEP_LL_COMBINE_PROFILE
        aggregate.SetProfile(profileOwner_);
#endif
        // Count DMA has retired. Auto uses the same final-stage scratch view as
        // M5, but owns its output events because M4 did not run on this launch.
        aggregate.InitDirectShared(
            arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes],
            args_->peerMems[args_->rank] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET, weightsAddr_,
            outputAddr_, bs_, h_, topK_, dtype_, core_ - sendCores_,
            recvCores_, workspace_, LlTransport::Rdma(args_));
        profile_.Mark(CombineProfile::FinalInitEnd);
        aggregate.Process();
        profile_.Mark(CombineProfile::FinalEnd);
        aggregate.DrainM6();
        profile_.Mark(CombineProfile::DrainEnd);
    }
    __aicore__ inline uint32_t AggregateSourceTokens()
    {
        ServerDedupSourceAggregate aggregate;
#ifdef DEEPEP_LL_COMBINE_PROFILE
        aggregate.SetProfile(profileOwner_);
#endif
        aggregate.Init(arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes],
                       args_->peerMems[args_->rank] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET,
                       masksAddr_, weightsAddr_, outputAddr_, bs_, h_, topK_, dtype_,
                       core_ - sendCores_, recvCores_, workspace_,
                       partialTasks_, enableDedup_, LlTransport::Rdma(args_));
        profile_.Mark(CombineProfile::FinalInitEnd);
        return aggregate.Process();
    }
    __aicore__ inline void DrainSharedOutputs(uint32_t retiredOutputs = 0U)
    {
        // M5 already consumed slots repurposed as inputs. Retire the remaining
        // physical slots exactly once, including unused seeds and M5 outputs.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane)
            if (((retiredOutputs >> lane) & 1U) == 0U) WaitFlag<HardEvent::MTE3_V>(lane);
    }
    template <class T>
    __aicore__ inline LocalTensor<T> Allocate(uint32_t bytes)
    {
        auto result = arena_.Get<uint8_t>()[usedBytes_].template ReinterpretCast<T>();
        usedBytes_ += bytes;
        return result;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const TEventID id = pipe_.FetchEventID(E);
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }
    __aicore__ inline GlobalTensor<float> Control(uint32_t rank, uint64_t offset)
    {
        GlobalTensor<float> result;
        result.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args_->peerMems[rank] + control_ + offset));
        if (LlTransport::Rdma(args_)) result.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        return result;
    }
    __aicore__ inline void SelectRows(int32_t type)
    {
        selected_ = 0U;
        if (tokens_ != 0U) {
            const uint64_t loadStart = profile_.DetailClock();
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();
            profile_.Elapsed(CombineProfile::StageSelectLoadTicks, loadStart);
            const uint64_t vectorStart = profile_.DetailClock();
            Compares(maskUb_, typeUb_, type, CMPMODE::EQ, tokens_);
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();
            profile_.Elapsed(CombineProfile::StageSelectVectorTicks, vectorStart);
        }
    }
    __aicore__ inline void CopyLocalPendingTokens()
    {
        SelectRows(3);
        profile_.Mark(CombineProfile::StageSelected);
        // Result lanes have no users in this phase. Keep zero gaps in a source
        // sized by packet blocks, not by the unrelated prepare-core row count.
        auto stageZeros = resultUb_.ReinterpretCast<float>();
        Duplicate<float>(stageZeros, 0.0f, blocks_ * 8U);
        Sync<HardEvent::V_MTE3>();
        const uint32_t base = uint32_t(selected_) / coreNum_, extra = uint32_t(selected_) % coreNum_;
        const uint32_t count = base + (core_ < extra ? 1U : 0U);
        const uint32_t start = core_ * base + (core_ < extra ? core_ : extra);
        profile_.Set(CombineProfile::StageRows, count);
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
        const uint32_t activeLanes = count < lanes_ ? count : lanes_;
#else
        const uint32_t activeLanes = lanes_;
#endif
        for (uint32_t lane = 0; lane < activeLanes; ++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
        uint32_t cacheBegin = 0U, cacheEnd = 0U;
        for (uint32_t task = 0; task < count; ++task) {
            const uint32_t row = uint32_t(packedUb_.GetValue(start + task));
            const uint64_t metaStart = profile_.DetailClock();
            CacheTableRows(row, start, task, count, cacheBegin, cacheEnd);
            profile_.Elapsed(CombineProfile::StageMetadataTicks, metaStart);
            const uint32_t slot = uint32_t(tableUb_.GetValue((row - cacheBegin) * 3U));
            const uint32_t lane = task % activeLanes;
            const uint64_t reuseStart = profile_.DetailClock();
            WaitFlag<HardEvent::MTE3_MTE2>(lane);
            profile_.Elapsed(CombineProfile::StageReuseWaitTicks, reuseStart);
            const uint64_t inputStart = profile_.DetailClock();
            auto payload = payloadUb_[lane * laneBytes_ / 2U];
            DataCopyPad(payload, expert_[uint64_t(row) * h_], {1U, h_ * 2U, 0U, 0U, 0U}, {false, 0U, 0U, uint16_t(0)});
            SetFlag<HardEvent::MTE2_MTE3>(lane);
            WaitFlag<HardEvent::MTE2_MTE3>(lane);
            profile_.Elapsed(CombineProfile::StageInputTicks, inputStart);
            const uint64_t storeStart = profile_.DetailClock();
            StoreStageWire(stage_[uint64_t(slot) * stageStride_ / 2U], payload, stageZeros, h_ * 2U);
            SetFlag<HardEvent::MTE3_MTE2>(lane);
            profile_.Elapsed(CombineProfile::StageStoreIssueTicks, storeStart);
        }
        profile_.Mark(CombineProfile::StageCopiesEnd);
        const uint64_t drainStart = profile_.DetailClock();
        // Also consumes seeded, unused lane events. All stage stores complete
        // before publication; individual tokens never wait immediately after store.
        for (uint32_t lane = 0; lane < activeLanes; ++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
        profile_.Elapsed(CombineProfile::StageDrainTicks, drainStart);
        profile_.Mark(CombineProfile::StageDrained);
        const uint64_t publishStart = profile_.DetailClock();
        Duplicate<float>(preparePublish_, 0.0f, 64U);
        Sync<HardEvent::V_S>();
        for (uint32_t j = 0; j < 8U; ++j) preparePublish_.SetValue(j * 8U, 1.0f);
        Sync<HardEvent::S_MTE3>();
        auto done = Control(args_->rank, COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET);
        DataCopyPad(done[core_ * 8U], preparePublish_, {8U, 32U, 0U, kCombinePrepareRowBytes - 32U, 0U});
        profile_.Elapsed(CombineProfile::StagePublishTicks, publishStart);
    }
    __aicore__ inline void CacheTableRows(uint32_t row, uint32_t start, uint32_t task, uint32_t count,
                                          uint32_t& cacheBegin, uint32_t& cacheEnd)
    {
        if (row >= cacheEnd) {
            cacheBegin = row;
            cacheEnd = row + 1U;
            for (uint32_t next = task + 1U; next < count; ++next) {
                const uint32_t nextRow = uint32_t(packedUb_.GetValue(start + next));
                if (nextRow - cacheBegin >= kEpServerDedupCombineTableRows) break;
                cacheEnd = nextRow + 1U;
            }
            // The old metadata has no users after its scalar offset is read.
            // Payload DMA owns separate lanes, so changing this cache needs
            // no payload drain and preserves the lane sequence across batches.
            DataCopyPad(tableUb_, table_[uint64_t(cacheBegin) * 3U], {1U, (cacheEnd - cacheBegin) * 12U, 0U, 0U, 0U},
                        {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void PrepareDirectRows()
    {
        selected_ = 0U;
        if (tokens_ == 0U) return;
        if (enableDedup_) {
            const uint64_t loadStart = profile_.DetailClock();
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();
            profile_.Elapsed(CombineProfile::DirectRouteLoadTicks, loadStart);
            const uint64_t vectorStart = profile_.DetailClock();
            Compares(maskUb_, typeUb_, int32_t(1), CMPMODE::EQ, (tokens_ + 63U) / 64U * 64U);
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();
            profile_.Elapsed(CombineProfile::DirectSelectVectorTicks, vectorStart);
        } else {
            // Every row is a direct row without the dedup token type table.
            CreateVecIndex(packedUb_, int32_t(0), tokens_);
            selected_ = tokens_;
            Sync<HardEvent::V_S>();
        }
        profile_.Mark(CombineProfile::DirectTypesEnd);
        if (selected_ == 0U) return;  // no routing reads for a type2/type3-only rank
        const uint32_t chunk = tokens_ < kEpServerDedupCombineTableRows ? tokens_ : kEpServerDedupCombineTableRows;
        CreateVecIndex(typeUb_, int32_t(0), chunk);
        PipeBarrier<PIPE_V>();
        Muls(typeUb_, typeUb_, int32_t(12), chunk);  // Table2 byte offsets
        for (uint32_t begin = 0U; begin < tokens_; begin += chunk) {
            const uint32_t span = tokens_ - begin < chunk ? tokens_ - begin : chunk;
            const uint64_t loadStart = profile_.DetailClock();
            Sync<HardEvent::V_MTE2>();  // table cache is no longer a Gather source
            DataCopyPad(tableUb_, table_[uint64_t(begin) * 3U], {1U, span * 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();
            profile_.Elapsed(CombineProfile::DirectRouteLoadTicks, loadStart);
            const uint64_t gatherStart = profile_.DetailClock();
            Gather(indexUb_[begin], tableUb_, typeUb_.ReinterpretCast<uint32_t>(), 0U, span);
            profile_.Elapsed(CombineProfile::DirectGatherTicks, gatherStart);
        }
        profile_.Mark(CombineProfile::DirectRoutesEnd);
        PipeBarrier<PIPE_V>();
        // Keep two persistent compact arrays: ranks in typeUb_, original rows
        // in packedUb_. indexUb_ becomes the per-server compact output list.
        if (enableDedup_) {
            GatherMask(typeUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
        } else {
            Muls(typeUb_, indexUb_, int32_t(1), tokens_);
        }
        Sync<HardEvent::V_S>();
        ServerDedupRankMask(maskUb_, tableUb_.ReinterpretCast<uint8_t>(), typeUb_, 0, int32_t(args_->rankSize),
                            uint32_t(selected_));
        uint64_t valid = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, valid);
        Sync<HardEvent::V_S>();
        if (valid != selected_) Trap();
    }
    __aicore__ inline uint32_t SelectDirectServer(uint32_t server)
    {
        ServerDedupRankMask(maskUb_, tableUb_.ReinterpretCast<uint8_t>(), typeUb_, int32_t(server * Group),
                            int32_t((server + 1U) * Group), uint32_t(selected_));
        uint64_t count = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, count);
        Sync<HardEvent::V_S>();
        return uint32_t(count);
    }
    __aicore__ inline void ReturnDirectTokens()
    {
        PrepareDirectRows();
        profile_.Mark(CombineProfile::DirectPrepared);
        if (selected_ == 0U) return;
        const uint64_t setupStart = profile_.DetailClock();
        if (LlTransport::Rdma(args_)) returnBatch_.Init(args_, resultStride_);
        const uint32_t servers = uint32_t(args_->rankSize) / Group;
        const uint32_t ownServer = uint32_t(args_->rank) / Group;
        uint32_t owner = 0U, sendSequence = 0U;
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
        const uint32_t owned = uint32_t(selected_) / sendCores_ + (core_ < uint32_t(selected_) % sendCores_ ? 1U : 0U);
        if (owned == 0U) return;
        const uint32_t activeLanes = owned < lanes_ ? owned : lanes_;
#else
        const uint32_t activeLanes = lanes_;
#endif
        uint64_t flagMask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < activeLanes; ++lane) {
            // The entire 32B Flag is initialized once and is not touched by Copy.
            Duplicate<float>(resultUb_[lane * resultStride_].ReinterpretCast<float>()[120U], 1.0f, flagMask,
                             uint8_t(blocks_), 1U, 16U);
            SetFlag<HardEvent::V_MTE2>(lane);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        PipeBarrier<PIPE_V>();
        profile_.Elapsed(CombineProfile::DirectSetupTicks, setupStart);
        profile_.Mark(CombineProfile::DirectSetupEnd);
        for (uint32_t step = 1U; step <= servers; ++step) {
            const uint64_t scanStart = profile_.DetailClock();
            const uint32_t count = SelectDirectServer(CombineReturnServer<Group>(servers, ownServer, step - 1U,
                LlTransport::Rdma(args_) && LlTransport::RanksPerHost(args_) == Group));
            profile_.Elapsed(CombineProfile::DirectScanTicks, scanStart);
            const uint32_t first = ServerDedupFirstOwned(core_, owner, sendCores_);
            owner = (owner + count) % sendCores_;
            for (uint32_t task = first; task < count; task += sendCores_) {
                const uint64_t metadataStart = profile_.DetailClock();
                const uint32_t row = uint32_t(indexUb_.GetValue(task));
                // Only owned rows reload a routing triple. The persistent rank
                // and original-row arrays are untouched between server rounds.
                DataCopyPad(tableUb_, table_[uint64_t(row) * 3U], {1U, 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
                Sync<HardEvent::MTE2_S>();
                const uint32_t source = uint32_t(tableUb_.GetValue(0U));
                const uint32_t token = uint32_t(tableUb_.GetValue(1U));
                const uint32_t topk = uint32_t(tableUb_.GetValue(2U));
                const uint32_t lane = sendSequence++ % activeLanes;
                profile_.Elapsed(CombineProfile::DirectMetadataTicks, metadataStart);
                // Input lifetime ends after Vector has consumed its last block.
                const uint64_t reuseStart = profile_.DetailClock();
                WaitFlag<HardEvent::V_MTE2>(lane);
                profile_.Elapsed(CombineProfile::DirectReuseWaitTicks, reuseStart);
                const uint64_t inputStart = profile_.DetailClock();
                auto input = payloadUb_[lane * laneBytes_ / 2U];
                DataCopyPad(input, expert_[uint64_t(row) * h_], {1U, h_ * 2U, 0U, 0U, 0U},
                            {false, 0U, 0U, uint16_t(0)});
                SetFlag<HardEvent::MTE2_V>(lane);
                profile_.Elapsed(CombineProfile::DirectInputIssueTicks, inputStart);
                const uint64_t inputWait = profile_.DetailClock();
                WaitFlag<HardEvent::MTE2_V>(lane);
                profile_.Elapsed(CombineProfile::DirectInputWaitTicks, inputWait);
                // Output lifetime ends only after the previous send has read it.
                const uint64_t outputStart = profile_.DetailClock();
                WaitFlag<HardEvent::MTE3_V>(lane);
                profile_.Elapsed(CombineProfile::DirectOutputWaitTicks, outputStart);
                const uint64_t packStart = profile_.DetailClock();
                auto output = resultUb_[lane * resultStride_];
                auto dstWords = output.ReinterpretCast<int32_t>();
                auto srcWords = input.ReinterpretCast<int32_t>();
                // Same nonquantized vector packing as Dispatch, without its tail.
                // Input UB covers blocks_*480B even though GM reads exactly 2H.
                Copy(dstWords, srcWords, uint64_t(64U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
                Copy(dstWords[64U], srcWords[64U], uint64_t(56U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
                SetFlag<HardEvent::V_MTE2>(lane);
                SetFlag<HardEvent::V_MTE3>(lane);
                WaitFlag<HardEvent::V_MTE3>(lane);

                profile_.Elapsed(CombineProfile::DirectPackTicks, packStart);
                // One destination owns this output. Its body is overwritten on
                // the next lane use, so SDK scratch need not be restored.
                const uint64_t offset = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET +
                    (uint64_t(token) * topK_ + topk) * resultStride_;
                profile_.MarkOnce(CombineProfile::DirectFirstSubmit);
                const uint64_t submitStart = profile_.DetailClock();
                SendReturnPacket(returnBatch_, args_, source, offset, output, resultStride_,
                                 CombineDeferReturn(activeLanes) && task + sendCores_ < count);
                profile_.SampleMax(CombineProfile::DirectSubmitTicks, CombineProfile::DirectSubmitMaxTicks,
                                   CombineProfile::DirectSubmitMaxPeer, CombineProfile::DirectSubmitMaxToken,
                                   submitStart, source, token);
                profile_.Add(CombineProfile::DirectSubmitCalls);
                profile_.Add(CombineProfile::DirectRemoteCalls, LlTransport::RemoteWrite(args_, source) ? 1U : 0U);
                profile_.Mark(CombineProfile::DirectLastSubmit);
                SetFlag<HardEvent::MTE3_V>(lane);
            }
            // No server fence: the next scan can overlap pending output DMA.
        }
        profile_.Set(CombineProfile::DirectRows, sendSequence);
        returnBatch_.SubmitPending();
        profile_.Mark(CombineProfile::DirectPendingEnd);
        const uint64_t drainStart = profile_.DetailClock();
        // Final retirement includes unused seeded lanes, never a per-token drain.
        for (uint32_t lane = 0; lane < activeLanes; ++lane) {
            WaitFlag<HardEvent::V_MTE2>(lane);
            WaitFlag<HardEvent::MTE3_V>(lane);
        }
        profile_.Elapsed(CombineProfile::DirectDrainTicks, drainStart);
    }
    // M4→M5交接需保留尚未完成的MTE3输出槽；只等待当前要复用的输入/向量资源。
    // 修改此处同步必须联查Partial与SourceAggregate的槽位生命周期，避免提前覆盖发送数据。
    __aicore__ inline void ReturnPartialTokens()
    {
        profile_.Mark(CombineProfile::PartialBegin);
        // M1 payload lanes are retired; complete this core's M2 publications
        // before reusing scratch. Independent control buffers remain resident.
        PipeBarrier<PIPE_ALL>();
        // Seed exactly once, including empty M4. M5 inherits these events.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane) SetFlag<HardEvent::MTE3_V>(lane);
        ServerDedupPartial<Group> partial;
#ifdef DEEPEP_LL_COMBINE_PROFILE
        partial.SetProfile(profileOwner_);
#endif
        partial.Init(args_, &pipe_, arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes], expertAddr_,
                     tableAddr_, relayAddr_, typesAddr_, tokens_, bs_, h_, topK_, experts_, dtype_, window_,
                     workspace_, &returnBatch_);
        profile_.Mark(CombineProfile::PartialInitEnd);
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
        // Metadata is already immutable. Plan while peers finish staging;
        // payload loads still begin strictly after the existing ready wait.
        profile_.Mark(CombineProfile::PlanBegin);
        partial.PrepareTasks();
        profile_.Mark(CombineProfile::PlanEnd);
        WaitServerReady();
#endif
        profile_.Mark(CombineProfile::PartialRunBegin);
        partialTasks_ = partial.Process();
        profile_.Mark(CombineProfile::PartialRunEnd);
        profile_.Set(CombineProfile::PartialRows, partialTasks_);
        // Retire only input and Vector users before scalar metadata reuse.
        // M4 output slots and MTE3_V events stay live for M5's per-slot waits.
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::V_S>();
        profile_.Mark(CombineProfile::PartialEnd);
    }
    __aicore__ inline void WaitFlags(GlobalTensor<float> gm, LocalTensor<float> input, LocalTensor<float> work,
                                     LocalTensor<float> sum, uint32_t flags, bool incoming)
    {
        while (true) {
            const uint64_t loadStart = profile_.DetailClock();
            // The previous reduction has completed before MTE2 overwrites input.
            Sync<HardEvent::V_MTE2>();
            DataCopyPad(input, gm, {1U, flags * 32U, 0U, 0U, 0U}, {false, 0U, 0U, 0.0f});
            Sync<HardEvent::MTE2_V>();
            profile_.Elapsed(incoming ? CombineProfile::ReadyPollLoadTicks : CombineProfile::NotifyPollLoadTicks,
                             loadStart);
            const uint64_t checkStart = profile_.DetailClock();
            ReduceSum<float>(sum, input, work, 1U, flags, 1U);
            Sync<HardEvent::V_S>();
            const bool ready = sum.GetValue(0U) == float(flags);
            profile_.Elapsed(incoming ? CombineProfile::ReadyPollCheckTicks : CombineProfile::NotifyPollCheckTicks,
                             checkStart);
            profile_.Add(incoming ? CombineProfile::ReadyPollCalls : CombineProfile::NotifyPollCalls);
            profile_.Add(incoming ? CombineProfile::ReadyPollMisses : CombineProfile::NotifyPollMisses,
                         ready ? 0U : 1U);
            if (ready) return;
        }
    }
    __aicore__ inline void NotifyServerCard()
    {
        profile_.Mark(CombineProfile::NotifyBegin);
        auto done = Control(args_->rank,
                            COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET + uint64_t(core_ - notifyBegin_) * kCombinePrepareRowBytes);
        WaitFlags(done, prepareInput_, prepareWork_, prepareSum_, coreNum_, false);
        profile_.Mark(CombineProfile::NotifyLocalReady);
        const uint64_t resetStart = profile_.DetailClock();
        Duplicate<float>(prepareZero_, 0.0f, coreNum_ * 8U);
        Duplicate<float>(notifyUb_, 0.0f, recvCores_ * 8U);
        Sync<HardEvent::V_S>();
        for (uint32_t r = 0; r < recvCores_; ++r) notifyUb_.SetValue(r * 8U, 1.0f);
        Sync<HardEvent::V_MTE3>();
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(done, prepareZero_, {1U, coreNum_ * 32U, 0U, 0U, 0U});
        profile_.Elapsed(CombineProfile::NotifyResetTicks, resetStart);
        const uint64_t publishStart = profile_.DetailClock();
        // Eight notification cores cover all peers in the logical group.
        // Ready rows retain 512-byte capacity, independent of active group size.
        const uint32_t groupBase = uint32_t(args_->rank) / Group *
                                   Group;
        for (uint32_t local = core_ - notifyBegin_; local < Group; local += 8U) {
            LlTransport::Records(args_, groupBase + local,
                control_ + COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET +
                (uint32_t(args_->rank) % Group) * 32U,
                notifyUb_.ReinterpretCast<uint8_t>(), recvCores_, 512U, false,
                prepareInput_.ReinterpretCast<uint8_t>());
        }
        profile_.Elapsed(CombineProfile::NotifyPublishTicks, publishStart);
        profile_.Mark(CombineProfile::NotifyEnd);
    }
    __aicore__ inline void WaitServerReady()
    {
        profile_.Mark(CombineProfile::ReadyBegin);
        auto ready = Control(args_->rank, COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET +
                                              uint64_t(core_ - sendCores_) * 512U);
        WaitFlags(ready, readyInput_, readyWork_, readySum_, Group, true);
        const uint64_t resetStart = profile_.DetailClock();
        Duplicate<float>(readyZero_, 0.0f, 128U);
        Sync<HardEvent::V_MTE3>();
        DataCopyPad(ready, readyZero_, {1U, 512U, 0U, 0U, 0U});
        profile_.Elapsed(CombineProfile::ReadyResetTicks, resetStart);
        profile_.Mark(CombineProfile::ReadyEnd);
    }

    CombineProfile::View profile_;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    CombineProfile::DeviceProfile* profileOwner_ = nullptr;
#endif
    EpCombineM45WorkspaceLayout workspace_{};
    LlTransport::DispatchRdmaBatch returnBatch_;
    uint32_t partialTasks_ = 0U;
    uint32_t coreNum_ = 0, sendCores_ = 0, recvCores_ = 0, notifyBegin_ = 0;
    TPipe pipe_;
    TBuf<> arena_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    GM_ADDR expertAddr_ = nullptr;
    GM_ADDR typesAddr_ = nullptr;
    GM_ADDR tableAddr_ = nullptr;
    GM_ADDR relayAddr_ = nullptr;
    GM_ADDR weightsAddr_ = nullptr;
    GM_ADDR masksAddr_ = nullptr;
    GM_ADDR outputAddr_ = nullptr;
    bool enableDedup_ = false;
    uint32_t bs_ = 0, experts_ = 0;
    uint8_t dtype_ = 0;
    uint32_t core_ = 0, h_ = 0, topK_ = 0, blocks_ = 0, resultStride_ = 0, tokens_ = 0, lanes_ = 0, laneBytes_ = 0,
             usedBytes_ = 0;
    uint64_t selected_ = 0, window_ = 0, control_ = 0, stageBase_ = 0, stageStride_ = 0;
    GlobalTensor<uint16_t> expert_, stage_;
    GlobalTensor<int32_t> table_, types_;
    LocalTensor<int32_t> typeUb_, indexUb_, packedUb_, tableUb_;
    LocalTensor<uint8_t> maskUb_, resultUb_;
    LocalTensor<uint16_t> payloadUb_;
    LocalTensor<float> preparePublish_, prepareInput_, prepareWork_, prepareSum_, prepareZero_;
    LocalTensor<float> readyInput_, readyWork_, readySum_, readyZero_, notifyUb_;
};
}  // namespace CombineDedup
#endif
