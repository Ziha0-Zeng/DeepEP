"""Execute the production routing plan against the original M4 scan on CPU."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "csrc/kernels/combine/low_latency"


def function_body(source, marker):
    start = source.index(marker)
    cursor = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


@pytest.mark.parametrize("receive_cores", [0, 8, 16, 24, 32, 36])
@pytest.mark.parametrize("remote_first", [False, True])
def test_production_plan_matches_original_scan(tmp_path, receive_cores, remote_first):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the C++ CPU check")
    # Reuse the checked event/Vector stubs. Only the two additional public
    # Vector primitives needed by M4 routing are supplied here.
    header = (ROOT / "tests/ll_rdma/metadata_stubs/kernel_operator.h").as_posix()
    (tmp_path / "kernel_operator.h").write_text('''#pragma once
#include "''' + header + '''"
namespace AscendC {
template<class T>void Muls(LocalTensor<T> d,LocalTensor<T> s,T x,uint32_t n){
 for(uint32_t i=0;i<n;++i)d.p[i]=s.p[i]*x;
}
template<class T>void Gather(LocalTensor<T> d,LocalTensor<T> s,LocalTensor<uint32_t> offsets,uint32_t base,uint32_t n){
 for(uint32_t i=0;i<n;++i)std::memcpy(d.p+i,reinterpret_cast<uint8_t*>(s.p)+base+offsets.p[i],sizeof(T));
}
}
''', encoding="utf-8")
    original = function_body((KERNEL / "combine_server_dedup_partial.h").read_text(encoding="utf-8"),
                             "__aicore__ inline uint32_t SelectServer(")
    source = r'''
#include <cassert>
#include <random>
#include <numeric>
#include "kernels/combine/low_latency/combine_partial_plan.h"
#include "kernels/combine/low_latency/combine_layout.h"
using namespace AscendC;
using namespace CombineDedup;
template<uint32_t Group>struct Original {
 CombineProfile::View profile_;
 struct Args{uint32_t rankSize;};Args a;Args* args_=&a;
 uint32_t recvCores_,tokens_,scanRows_;
 LocalTensor<uint8_t> scan_;LocalTensor<int32_t> rows_;
 GlobalTensor<int32_t> types_,table2_;
 template<HardEvent E>void Sync(){SetFlag<E>(0U);WaitFlag<E>(0U);}
''' + original + r'''
};
template<uint32_t Group>void Check(uint32_t world,uint32_t rank,uint32_t c,uint32_t n,uint32_t h,uint32_t k,uint32_t pattern,bool crossHost=false){
 cores=c;const uint32_t recv=EpCombineRecvCoreCount(c);if(!recv)return;
 std::mt19937 rng(n*137+world+pattern);
 std::vector<int32_t> types(n),routes(n*3),hits(n,0);
 for(uint32_t i=0;i<n;++i){
  types[i]=pattern==1?2:pattern==2?3:int32_t(1+rng()%3);
  routes[i*3]=types[i]==2?int32_t(pattern==3?world-1:rng()%world):-1;
  routes[i*3+1]=i%72;routes[i*3+2]=i%k;
 }
 const auto layout=EpCombineM45Layout(n,72,h,k,recv);
 assert(layout.inputLanes && layout.outputLanes);
 const uint32_t bytes=layout.phaseOffset,ownedBytes=EpCombineM4TaskBytes(n,recv);
 for(uint32_t worker=0;worker<recv;++worker){
  core=c-recv+worker;
  std::vector<uint64_t> arena((bytes+ownedBytes+64)/8,0x5b5b5b5b5b5b5b5bULL);
  auto raw=reinterpret_cast<uint8_t*>(arena.data());
  LocalTensor<int32_t> owned{reinterpret_cast<int32_t*>(raw+bytes)};
  PartialTaskPlan<Group> plan;
  // M4 output events are already seeded by the caller; planning must leave
  // them untouched while using its own event directions.
  for(uint32_t lane=0;lane<8;++lane)SetFlag<HardEvent::MTE3_V>(lane);
  bool ready=plan.Prepare({raw},bytes,owned,{types.data()},{routes.data()},n,world,rank,worker,recv,crossHost);
  for(uint32_t lane=0;lane<8;++lane)WaitFlag<HardEvent::MTE3_V>(lane);
  for(auto& direction:events)for(bool live:direction)assert(!live);
  assert(ready==(EpCombineM4ScanBytes(EpCombineAlign(n,64U))<=bytes));
  if(!ready){for(auto word:arena)assert(word==0x5b5b5b5b5b5b5b5bULL);continue;}
  std::vector<int32_t> expected;
  const uint32_t scanRows=EpCombineM4ScanRows(h,k);
  std::vector<uint64_t> scan((EpCombineM4ScanBytes(scanRows)+64)/8,0);
  std::vector<int32_t> selected((n+recv-1)/recv+8,-99);
  Original<Group> old{{},{world},nullptr,recv,n,scanRows,{reinterpret_cast<uint8_t*>(scan.data())},
                       {selected.data()},{types.data()},{routes.data()}};
  old.args_=&old.a;
  uint32_t owner=0;
  for(uint32_t step=0;step<world/Group;++step){
   uint32_t group=CombineReturnServer<Group>(world/Group,rank/Group,step,crossHost);
   uint32_t count=old.SelectServer(group,owner,step==0);
   expected.insert(expected.end(),selected.begin(),selected.begin()+count);
   assert(plan.End(step)==expected.size());
  }
  for(uint32_t i=0;i<expected.size();++i){assert(owned.GetValue(i)==expected[i]);++hits[expected[i]];}
  assert(expected.size()<=(n+recv-1)/recv);
  for(uint32_t i=bytes+expected.size()*4;i<bytes+ownedBytes+64;++i)assert(raw[i]==0x5b);
  // The temporary routing snapshot becomes FP32 compute scratch. The owned
  // list must survive that overwrite until the last M4 task has been sent.
  std::memset(raw,0xa5,bytes);
  for(uint32_t i=0;i<expected.size();++i)assert(owned.GetValue(i)==expected[i]);
 }
 if(EpCombineM4ScanBytes(EpCombineAlign(n,64U))<=bytes)
  for(uint32_t i=0;i<n;++i)assert(hits[i]==(types[i]==2?1:0));
}
void BadRank(){
 std::vector<uint64_t> scratch(4096),output(32,0);
 std::vector<int32_t> types{3,2,1},routes{99999,0,0,-1,0,0,99999,0,0};
 for(int32_t bad:{-1,16}){
  routes[3]=bad;PartialTaskPlan<4> plan;bool trapped=false;
  try{plan.Prepare({reinterpret_cast<uint8_t*>(scratch.data())},32768,{reinterpret_cast<int32_t*>(output.data())},
       {types.data()},{routes.data()},3,16,0,0,8);}catch(const std::runtime_error&){trapped=true;}
  assert(trapped);
 }
 routes[3]=15;PartialTaskPlan<4> plan;
 assert(plan.Prepare({reinterpret_cast<uint8_t*>(scratch.data())},32768,{reinterpret_cast<int32_t*>(output.data())},
       {types.data()},{routes.data()},3,16,0,0,8));
}
int main(){
 // A 56-AIV device must admit 20 send + 36 receive, as well as the
 // original 28/28 split. Check Host admission before executing ownership.
 assert(EpCombineRecvCoreCount(56)==(DEEPEP_LL_COMBINE_RECV_CORES?DEEPEP_LL_COMBINE_RECV_CORES:28));
 if(DEEPEP_LL_COMBINE_RECV_CORES)assert(EpCombineRecvCoreCount(DEEPEP_LL_COMBINE_RECV_CORES)==0);
 for(uint32_t world:{4,16,32})for(uint32_t bs:{32,48,64,72}){
  Layout launch;assert(BuildLayout(world,bs,3584,8,384,launch,56));
  const auto ub=EpCombineM45Layout(10000,bs,3584,8,EpCombineRecvCoreCount(56));
  assert(ub.inputLanes && ub.totalBytes<=kEpServerDedupCombineRecvUbBytes);
 }
 for(uint32_t c:{16,32,56,64,72})for(uint32_t n:{0,1,7,63,64,65,255,256,257,576,577,1024,10000})
  for(uint32_t pattern=0;pattern<4;++pattern){
   Check<4>(16,pattern*4,c,n,3584,8,pattern);
   Check<2>(2,pattern%2,c,n,3584,8,pattern);
   Check<4>(4,pattern,c,n,3584,8,pattern);
   Check<4>(8,pattern,c,n,3584,8,pattern);
   Check<4>(8,pattern,c,n,3584,8,pattern,true);
   Check<4>(8,pattern+4,c,n,3584,8,pattern,true);
   Check<16>(128,pattern*31,c,n,7168,16,pattern);
  }
 for(uint32_t world:{32,64,256})for(uint32_t n:{0,1,65,576})
  Check<16>(world,world-1,56,n,3584,6,0);
 for(uint32_t h:{256,512,7168})for(uint32_t n:{1,7,65,576})Check<4>(16,15,32,n,h,8,0);
 BadRank();
 std::puts("PASS production partial plan: ordering, ownership, guards, fallback and invalid sources");
}
'''
    cpp = tmp_path / "partial-plan.cpp"
    cpp.write_text(source, encoding="utf-8")
    exe = tmp_path / "partial-plan.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", f"-DDEEPEP_LL_COMBINE_RECV_CORES={receive_cores}",
                    *(["-DDEEPEP_LL_COMBINE_REMOTE_FIRST=1"] if remote_first else []), "-I" + str(tmp_path),
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=90)
    assert "PASS production partial plan" in result.stdout


def test_variant_has_separate_build_cache_and_explicit_cli():
    spec = importlib.util.spec_from_file_location("combine_build_support", ROOT / "build_support.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    baseline = {"ll_combine_small_batch": False}
    optimized = {"ll_combine_small_batch": True}
    assert build.build_cache_key(baseline) != build.build_cache_key(optimized)
    import runpy
    st = runpy.run_path(str(ROOT / "tests/elastic/test_ep_low_latency.py"))
    for variant in ("baseline", "optimized"):
        args = st["parse_args"](["--combine-variant", variant, "--num-processes", "16",
                                "--num-tokens", "1", "72", "--num-max-tokens-per-rank", "256"])
        assert args.combine_variant == variant and st["validate_args"](args, 16) == 256


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("returns", [False, True])
def test_production_lane_events_and_payloads(tmp_path, optimized, returns):
    """Execute both edited loops with deferred stores and checked event IDs."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the C++ CPU check")
    kernel = (KERNEL / "combine_server_dedup_kernel.h").read_text(encoding="utf-8")
    methods = "\n".join(function_body(kernel, "__aicore__ inline void " + name + "()")
                        for name in ("CopyLocalPendingTokens", "ReturnDirectTokens"))
    header = (ROOT / "tests/ll_rdma/metadata_stubs/kernel_operator.h").as_posix()
    source = r'''
#define DEEPEP_TEST_DEFERRED_MTE3
#include "''' + header + r'''"
#include <cassert>
#include <algorithm>
#include <numeric>
#include "kernels/combine/low_latency/combine_server_dedup_select.h"
#include "kernels/combine/low_latency/combine_profile_device.h"
#include "kernels/combine/low_latency/combine_return_schedule.h"
using namespace AscendC;
using namespace CombineDedup;
struct Args{uint32_t rank=0,rankSize=16;bool rdma=false;};
namespace LlTransport {bool Rdma(Args* a){return a->rdma;} uint32_t RanksPerHost(Args*){return 16U;} bool RemoteWrite(Args* a,uint32_t p){return a->rdma && p>=16;}}
constexpr uint32_t kCombinePrepareRowBytes=72*32;
constexpr uint64_t COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET=0;
constexpr uint64_t COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET=0;
struct Batch{void Init(Args*,uint32_t){}void SubmitPending(){}};
inline uint32_t sent=0,staged=0;
inline std::vector<uint8_t> returns;
void StoreStageWire(GlobalTensor<uint16_t> d,LocalTensor<uint16_t> s,LocalTensor<float>,uint32_t bytes){
 ++staged;DataCopy(d,s,bytes/2);
}
struct CopyParams{uint32_t dstStride,srcStride,dstRepeat,srcRepeat;};
void Copy(LocalTensor<int32_t> d,LocalTensor<int32_t> s,uint64_t words,uint8_t reps,CopyParams p){
 for(uint32_t r=0;r<reps;++r)std::memcpy(d.p+r*p.dstRepeat*8,s.p+r*p.srcRepeat*8,words*4);
}
void SendReturnPacket(Batch&,Args*,uint32_t,uint64_t off,LocalTensor<uint8_t> u,uint32_t bytes,bool){
 ++sent;DataCopy(GlobalTensor<uint8_t>{returns.data()+off},u,bytes);
}
template<uint32_t Group>struct Test{
 CombineProfile::View profile_;
 Args a;Args* args_=&a;Batch returnBatch_;
 uint32_t core_,coreNum_,sendCores_,lanes_,selected_,h_=256,blocks_=2,laneBytes_=960,resultStride_=1024;
 uint32_t topK_=1;uint64_t stageStride_=1024,window_=0;
 std::vector<int32_t> rows,route,table,indices;
 std::vector<uint16_t> experts,payload,stage;
 std::vector<uint8_t> output;
 std::vector<float> publication,done;
 LocalTensor<int32_t> packedUb_,tableUb_,indexUb_;GlobalTensor<int32_t> table_;
 LocalTensor<uint16_t> payloadUb_;GlobalTensor<uint16_t> expert_,stage_;
 LocalTensor<uint8_t> resultUb_;LocalTensor<float> preparePublish_;
 Test(uint32_t n,uint32_t cores,uint32_t c,uint32_t lanes,uint32_t world,bool rdma):
  core_(c),coreNum_(cores),sendCores_(cores/2),lanes_(lanes),selected_(n),rows(n),route(n*3),table(n*3+8),indices(n),
  experts(n*h_),payload(lanes*laneBytes_/2),stage(n*stageStride_/2,0xdead),output(lanes*resultStride_,0x69),publication(64),done(8*72*8,0){
  a.rankSize=world;a.rdma=rdma;std::iota(rows.begin(),rows.end(),0);
  for(uint32_t r=0;r<n;++r){route[3*r]=(r*17+7)%world;route[3*r+1]=r;
   for(uint32_t j=0;j<h_;++j)experts[r*h_+j]=uint16_t(r*37+j);}
  packedUb_={rows.data()};tableUb_={table.data()};indexUb_={indices.data()};table_={route.data()};
  payloadUb_={payload.data()};expert_={experts.data()};stage_={stage.data()};resultUb_={output.data()};preparePublish_={publication.data()};
 }
 template<HardEvent E>void Sync(){SetFlag<E>(0);WaitFlag<E>(0);}
 void SelectRows(int){}
 void CacheTableRows(uint32_t row,uint32_t,uint32_t,uint32_t,uint32_t& begin,uint32_t& end){begin=row;end=row+1;table[0]=row;}
 GlobalTensor<float> Control(uint32_t,uint64_t){assert(deferredMte3.empty());return {done.data()};}
 void PrepareDirectRows(){}
 uint32_t SelectDirectServer(uint32_t server){uint32_t n=0;for(uint32_t r=0;r<selected_;++r)
  if(uint32_t(route[3*r])/Group==server)indices[n++]=r;return n;}
''' + methods + r'''
};
void Retired(){assert(deferredMte3.empty());for(auto& dir:events)for(bool live:dir)assert(!live);}
template<uint32_t Group>void Check(uint32_t world,uint32_t cores,uint32_t lanes,uint32_t n,bool rdma){
 for(uint32_t c=0;c<cores;++c){
  Test<Group> t(n,cores,c,lanes,world,rdma);staged=0;t.CopyLocalPendingTokens();Retired();
  uint32_t count=n/cores+(c<n%cores),start=c*(n/cores)+std::min(c,n%cores);assert(staged==count);
  for(uint32_t r=0;r<n;++r)for(uint32_t j=0;j<t.h_;++j)
   assert(t.stage[r*t.stageStride_/2+j]==(r>=start&&r<start+count?t.experts[r*t.h_+j]:uint16_t(0xdead)));
  for(uint32_t j=0;j<8;++j)assert(t.done[j*72*8+c*8]==1.0f);
 }
 returns.assign(n*1024,0x5b);sent=0;
 for(uint32_t c=0;c<cores/2;++c){Test<Group> t(n,cores,c,lanes,world,rdma);t.ReturnDirectTokens();Retired();}
 assert(sent==n);
 for(uint32_t r=0;r<n;++r){
  for(uint32_t j=0;j<256;++j){uint16_t value;uint32_t byte=j*2,wire=byte/480*512+byte%480;
   std::memcpy(&value,returns.data()+r*1024+wire,2);assert(value==uint16_t(r*37+j));}
  for(uint32_t b=0;b<2;++b)for(uint32_t f=0;f<8;++f){float value;std::memcpy(&value,returns.data()+r*1024+b*512+480+f*4,4);assert(value==1.0f);}
 }
}
int main(){for(uint32_t c:{16,32,56,64,72})for(uint32_t lanes:{1,2,8})for(uint32_t n:{0,1,7,17,71,72,255}){
 Check<4>(16,c,lanes,n,false);Check<16>(128,c,lanes,n,true);
}}
'''
    cpp, exe = tmp_path / "combine-lanes.cpp", tmp_path / "combine-lanes.exe"
    cpp.write_text(source, encoding="utf-8")
    flags = ["-DDEEPEP_LL_COMBINE_SMALL_BATCH=1"] if optimized else []
    if returns:
        flags += ["-DDEEPEP_LL_COMBINE_REMOTE_FIRST=1", "-DDEEPEP_LL_COMBINE_DEFER_RETURN=1"]
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", *flags,
                    "-I" + str(ROOT / "csrc"), "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    str(cpp), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)
