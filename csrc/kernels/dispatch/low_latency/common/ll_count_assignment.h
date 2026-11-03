#pragma once
#include <cstdint>
#include "ll_group_policy.h"

namespace DispatchDedup {
struct LlCountAssignment {
    uint32_t localDomain, localPeers, alignedRecords, reservedRecords;
    uint32_t remoteRank, remoteRecords;
};
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
uint32_t CountRemotePeer(uint32_t rank, uint32_t world, uint32_t task,
                         uint32_t ranksPerHost = kLlPhysicalRanksPerHost)
{
    // Compact remote-rank enumeration skips this source's physical host.
    const uint32_t hostFirst = rank / ranksPerHost * ranksPerHost;
    return task < world - ranksPerHost
        ? (task < hostFirst ? task : task + ranksPerHost) : world;
}
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
uint32_t CountLocalPeer(uint32_t rank, uint32_t task, bool directRanks,
                        uint32_t ranksPerHost = kLlPhysicalRanksPerHost)
{
    // MTE-only domains retain coverage of all ranks; mixed NDR uses this host.
    return (directRanks ? rank / ranksPerHost * ranksPerHost : 0U) + task;
}
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
LlCountAssignment CountAssignment(uint32_t rank, uint32_t world, uint32_t experts,
                                   uint32_t cores, uint32_t core, bool directRanks,
                                   uint32_t ranksPerHost = kLlPhysicalRanksPerHost)
{
    const uint32_t domain = directRanks ? ranksPerHost : world;
    const uint32_t peers = core < domain ? (domain - 1U - core) / cores + 1U : 0U;
    const uint32_t records = peers ? (experts / world + 7U) / 8U * 8U : 0U;
    const uint32_t remote = directRanks ? CountRemotePeer(rank, world, core, ranksPerHost) : world;
    const uint32_t remoteRecords = remote < world ? (experts / world + 1U + 7U) / 8U * 8U : 0U;
    return {domain, peers, records, peers * records, remote, remoteRecords};
}
}
