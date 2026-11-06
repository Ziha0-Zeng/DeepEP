#pragma once
#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"

namespace opsbench {
// A handle contains a PE number, not an HCCL channel. The upstream batch
// boundary is retained; SHMEM posts each Put and quiets at that boundary.
struct Batch {
    uint32_t peer;
    bool pending = false;
};
class Transport {
    AscendC::LocalTensor<uint8_t> scratch_;

public:
    __aicore__ inline void Init(AscendC::LocalTensor<uint8_t> scratch, uint32_t bytes)
    {
        if (bytes < 128U) AscendC::Trap();
        scratch_ = scratch;
    }
    __aicore__ inline Batch MakeBatchHandle(uint64_t peer, AscendC::LocalTensor<uint8_t>, uint32_t, GM_ADDR, GM_ADDR)
    {
        return {static_cast<uint32_t>(peer), false};
    }
    __aicore__ inline void WriteNbi(Batch& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        // Both pointers are local symmetric addresses. SHMEM translates dst.
        aclshmemx_roce_qp_put_nbi(dst, src, (__ubuf__ uint8_t*)scratch_.GetPhyAddr(), bytes, batch.peer,
                                  AscendC::GetBlockIdx(), 7U);
        batch.pending = true;
    }
    __aicore__ inline void BatchCommit(Batch& batch)
    {
        if (batch.pending) {
            aclshmemx_roce_qp_quiet(batch.peer, AscendC::GetBlockIdx(), (__ubuf__ uint8_t*)scratch_.GetPhyAddr(), 7U);
            batch.pending = false;
        }
    }
};
}  // namespace opsbench
