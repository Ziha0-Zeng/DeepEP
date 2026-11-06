"""Execute the real suite runner with CPU substitutes for MPI, ACL and modules."""
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_shared_session_lifecycle(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for host C++ checks")

    def write(name, text):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    write("state.hpp", r'''
#pragma once
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
inline int mpiInit=0, mpiEnd=0, aclInitCount=0, aclEnd=0, sessionInit=0, sessionEnd=0;
inline int resets=0, aborts=0, calls=0, loads=0, failPhase=-1;
inline bool failFinalize=false, busy=false;
inline unsigned char memory[128];
inline std::vector<std::string> order;
inline int test_setenv(const char*, const char*, int) { return 0; }
''')
    write("mpi.h", r'''
#pragma once
#include "state.hpp"
using MPI_Comm=int;
constexpr int MPI_SUCCESS=0, MPI_COMM_WORLD=1, MPI_COMM_NULL=0, MPI_ERRORS_RETURN=0;
constexpr int MPI_COMM_TYPE_SHARED=0, MPI_INFO_NULL=0, MPI_INT=4, MPI_BYTE=1, MPI_UINT64_T=8;
constexpr int MPI_MAX_ERROR_STRING=1024;
inline int MPI_Init(int*,char***) { ++mpiInit; return 0; }
inline int MPI_Finalize() { ++mpiEnd; return 0; }
inline int MPI_Abort(int,int) { ++aborts; return 0; }
inline int MPI_Comm_set_errhandler(int,int) { return 0; }
inline int MPI_Comm_rank(int,int* p) { *p=0; return 0; }
inline int MPI_Comm_size(int,int* p) { *p=4; return 0; }
inline int MPI_Comm_split_type(int,int,int,int,int* p) { *p=2; return 0; }
inline int MPI_Comm_free(int*) { return 0; }
inline int MPI_Error_string(int,char*,int* n) { *n=0; return 0; }
inline int MPI_Bcast(void*,int,int,int,int) { return 0; }
inline int MPI_Barrier(int) { assert(!busy); return 0; }
inline int MPI_Allgather(const void* p,int n,int type,void* out,int,int,int) {
    for(int i=0;i<4;++i) std::memcpy(static_cast<char*>(out)+i*n*type,p,n*type);
    return 0;
}
''')
    write("acl/acl.h", r'''
#pragma once
#include "state.hpp"
using aclrtStream=void*;
inline int aclInit(const char*) { ++aclInitCount; return 0; }
inline int aclFinalize() { ++aclEnd; return 0; }
inline int aclrtSetDevice(int) { return 0; }
inline int aclrtCreateStream(aclrtStream* p) { *p=memory; return 0; }
inline int aclrtDestroyStream(aclrtStream) { assert(!busy); return 0; }
inline int aclrtResetDevice(int) { return 0; }
inline int aclrtSynchronizeStream(aclrtStream) { busy=false; return 0; }
inline int aclrtMemset(void* p,size_t capacity,int value,size_t n) {
    assert(!busy && p==memory && capacity==sizeof(memory) && n==capacity);
    ++resets; std::memset(p,value,n); return 0;
}
''')
    write("runtime/session.hpp", r'''
#pragma once
#include "state.hpp"
#include <stdexcept>
namespace deepep {
struct Session {
    Session(int,int,uint64_t bytes,const char*,int,bool,bool,bool) {
        assert(bytes==(8ULL<<30)); ++sessionInit; std::memset(memory,0,sizeof(memory));
    }
    void* workspace() { return memory; }
    void* ll_rdma_stage() { return memory+64; }
    uint64_t capacity() { return 8ULL<<30; }
    uint64_t workspace_allocation_bytes() { return sizeof(memory); }
    uint32_t ll_rdma_qps() { return 56; }
    void close(bool) {
        assert(!busy); ++sessionEnd;
        if(failFinalize) throw std::runtime_error("injected finalize failure");
    }
};
}
''')
    write("runtime/ll_core_config.hpp", r'''
#pragma once
namespace deepep {
struct CoreConfig { unsigned active; };
inline CoreConfig query_ll_core_config() { return {56}; }
}
''')
    write("host/data_plane/shmem_host_p2p_sync.h", "#pragma once\ninline unsigned long long shmemx_get_ffts_config() {return 42;}\n")
    write("dlfcn.h", r'''
#pragma once
#include "state.hpp"
#include "variant_api.hpp"
constexpr int RTLD_NOW=2, RTLD_LOCAL=0;
inline void invoke(const llbench::Runtime* r,const char* name) {
    assert(!busy && r->ffts==42 && r->rdmaQps==56 && r->rdmaStage==memory+64);
    for(auto byte:memory) assert(byte==0);
    order.push_back(name);
    if(calls++==failPhase) throw std::runtime_error("injected phase failure");
    std::filesystem::create_directories(r->options.output);
    std::ofstream(std::filesystem::path(r->options.output)/"rank-0.csv") << "samples\n";
    std::memset(memory,0x55,sizeof(memory)); busy=true;
}
inline void base(const llbench::Runtime* r) {invoke(r,"base");}
inline void opt(const llbench::Runtime* r) {invoke(r,"opt");}
inline void* dlopen(const std::filesystem::path::value_type* name,int flags) {
    assert(flags==(RTLD_NOW|RTLD_LOCAL)); ++loads;
    return reinterpret_cast<void*>(std::filesystem::path(name).parent_path().filename()=="ll-cpp-base" ? 1:2);
}
inline void* dlsym(void* handle,const char* name) {
    assert(std::string(name)=="ll_bench_run");
    return reinterpret_cast<void*>(handle==reinterpret_cast<void*>(1) ? base:opt);
}
inline const char* dlerror() {return "mock error";}
inline int dlclose(void*) {assert(false); return 0;}
''')
    write("test.cpp", r'''
#include "state.hpp"
#define main production_main
#define setenv test_setenv
#include "runner.cpp"
#undef main
#undef setenv
int main(int argc,char** argv) {
    assert(argc==3);
    std::filesystem::path root=argv[1];
    const std::string mode=argv[2];
    failPhase=mode=="phase-failure" ? 1:-1;
    failFinalize=mode=="finalize-failure";
    std::vector<std::string> args={"bench","--variants","base,opt,base",
        "--build-root",(root/"build").string(),"--output-dir",(root/"results").string()};
    std::vector<char*> pointers;
    for(auto& arg:args) pointers.push_back(arg.data());
    int rc=run_suite(int(pointers.size()),pointers.data(),root/"build/ll-cpp-base/ll_combine_bench");
    assert(mpiInit==1 && aclInitCount==1 && sessionInit==1 && loads==2);
    const bool ok=mode=="success";
    assert((rc==0)==ok && aborts==!ok && mpiEnd==ok && aclEnd==ok);
    assert(resets==(failPhase<0 ? 2:1));
    assert(sessionEnd==(failPhase<0));
    assert(order==(failPhase<0 ? std::vector<std::string>{"base","opt","base"}:
                               std::vector<std::string>{"base","opt"}));
    for(int i=0;i<3;++i) {
        const auto folder=root/"results"/(std::to_string(i)+(i==1?"-opt":"-base"));
        assert(std::filesystem::exists(folder/"rank-0.PASS")==ok);
        assert(std::filesystem::exists(folder/"rank-0.SAMPLED")== (failPhase<0 || i<failPhase));
    }
}
''')
    executable = tmp_path / "test.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-I", str(tmp_path), "-I", str(HERE),
                    str(tmp_path / "test.cpp"), "-o", str(executable)], check=True, timeout=180)
    for mode in ("success", "phase-failure", "finalize-failure"):
        subprocess.run([str(executable), str(tmp_path / mode), mode], check=True, timeout=30)
