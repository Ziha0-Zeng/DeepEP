"""Execute the actual SIMT WQE builder on CPU; no device timing claims."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def compile_cpp(tmp_path, text):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    source = tmp_path / "check.cpp"
    source.write_text(text, encoding="utf8")
    binary = tmp_path / ("check.exe" if os.name == "nt" else "check")
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-I" + str(ROOT / "csrc/direct"), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=240,
    )
    return binary


def run_cpp(binary, **kwargs):
    try:
        return subprocess.run([str(binary)], capture_output=True, timeout=60, **kwargs)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("C++ harness compiled, but Windows application control blocks execution (4551)")
        raise


def test_real_wqe_builders_and_address_mapping(tmp_path):
    source = (ROOT / "csrc/direct/kernels/combine/udma_combine_sender.h").read_text()
    body = source[
        source.index("template <uint32_t BatchWqes, bool Direct>") : source.index("template <uint32_t Capacity =")
    ]
    harness = (
        r"""
#include <cstdint>
#include <cassert>
#include <vector>
#include <random>
#include <stdexcept>
#define __simt_vf__
#define __launch_bounds__(...)
#define __ubuf__
#define __gm__
struct { uint32_t x; } threadIdx;
namespace __asc_simt_vf { void __trap() { throw std::runtime_error("trap"); } }
"""
        + body
        + r"""
template<unsigned N> void check() {
    std::mt19937 rng(42 + N);
    for (unsigned trial=0; trial<300; ++trial) {
        const unsigned fields = 1 + trial%2, count = 1+rng()%N;
        const unsigned step = 1+trial%3, index_step=1+trial%2;
        std::vector<uint64_t> spans(N*4), headers(128*16), bases(128);
        std::vector<std::vector<int32_t>> indices(N, std::vector<int32_t>(1024));
        std::vector<uint64_t> expected(N*8, 0xdead), a=expected, b=expected;
        for (unsigned j=0;j<headers.size();++j) headers[j]=100000+j;
        for (unsigned j=0;j<bases.size();++j) bases[j]=0x10000000ull+j*0x10000000ull;
        unsigned begin=0, nspans=0;
        while (begin<count) {
            unsigned n=trial%3==0 ? count-begin : 1+rng()%(count-begin);
            unsigned peer=rng()%128, row=100+nspans*1024, field=rng()%fields;
            auto &idx=indices[nspans];
            for (auto &v:idx) v=rng()%4096;
            auto *span=spans.data()+4*nspans;
            span[0]=reinterpret_cast<uint64_t>(idx.data());
            span[1]=uint64_t(row)|(uint64_t(n)<<32);
            span[2]=uint64_t(peer)|(uint64_t(field)<<32);
            span[3]=begin;
            for(unsigned j=0;j<n;++j) {
                unsigned i=begin+j, pos=field+j;
                bool weight=(pos%fields)!=0;
                uint64_t token=idx[(pos/fields)*index_step];
                for(unsigned h=0;h<5;++h) expected[i*8+h]=headers[(peer*2+(i+1==count))*8+h];
                expected[i*8+5]=bases[peer]+(weight?0x200000ull:0x100000ull)+token*(weight?32:5632);
                expected[i*8+6]=weight?32:5632;
                expected[i*8+7]=(weight?0x40000000ull:0x80000000ull)+(row+(pos/fields)*step)*(weight?32ull:5632ull);
            }
            begin+=n; ++nspans;
        }
        for(unsigned i=0;i<N;++i) {
            threadIdx.x=i;
            BuildCombineWqes<N,false>(a.data(),spans.data(),headers.data(),bases.data(),
                0x80000000ull,0x40000000ull,0x100000ull,0x200000ull,5632,32,5632,32,4096,
                fields,nspans,count,step,index_step,0,0,0);
            BuildCombineWqes<N,true>(b.data(),spans.data(),headers.data(),bases.data(),
                0x80000000ull,0x40000000ull,0x100000ull,0x200000ull,5632,32,5632,32,4096,
                fields,nspans,count,step,index_step,0,0,0);
        }
        assert(a==expected && b==expected);
        // Zero-work threads must not read any descriptor or write output.
        threadIdx.x=0;
        BuildCombineWqes<N,true>(nullptr,nullptr,nullptr,nullptr,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0);
        indices[0][0]=-1;
        threadIdx.x=0;
        bool trapped=false;
        try { BuildCombineWqes<N,true>(b.data(),spans.data(),headers.data(),bases.data(),
                0,0,0,0,5632,32,5632,32,4096,fields,nspans,count,step,index_step,0,0,0); }
        catch (const std::runtime_error&) { trapped=true; }
        assert(trapped);
    }
}
int main() { check<32>(); check<64>(); check<128>(); check<256>(); }
"""
    )
    run_cpp(compile_cpp(tmp_path, harness), check=True)


def test_native_option_validation(tmp_path):
    binary = compile_cpp(
        tmp_path,
        r"""
#include "common/combine_wqe_options.h"
#include <cstdio>
int main() {
    try { const auto c=ascend_deepep::udma::ReadCombineWqeOptions();
          std::printf("%u %u", c.batch,c.direct); }
    catch(const std::invalid_argument&) { return 2; }
}
""",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("DEEPEP_CLOS_COMBINE_WQE_")}
    assert run_cpp(binary, env=env, check=True).stdout.decode() == "32 0"
    for batch in (32, 64, 128, 256):
        for builder in ("legacy", "direct"):
            e = env | {"DEEPEP_CLOS_COMBINE_WQE_BATCH": str(batch), "DEEPEP_CLOS_COMBINE_WQE_BUILDER": builder}
            assert run_cpp(binary, env=e, check=True).stdout.decode() == f"{batch} {int(builder == 'direct')}"
    for name, value in (("BATCH", "0"), ("BATCH", "33"), ("BATCH", "256junk"), ("BUILDER", "DIRECT")):
        assert run_cpp(binary, env=env | {f"DEEPEP_CLOS_COMBINE_WQE_{name}": value}).returncode == 2


def test_sweep_plan():
    output = subprocess.check_output([sys.executable, str(ROOT / "benchmarks/clos/bench_wqe_sweep.py"), "--plan-only"])
    plan = json.loads(output)["plans"][0]
    assert plan["shape"] == dict(tokens=4096, hidden=2816, topk=8, aiv=8)
    assert len(plan["operations"]["cached_dispatch"]) == 1
    for op in ("combine_unweighted", "combine_weighted"):
        cs = plan["operations"][op]
        assert len(cs) == 8 and len({c["id"] for c in cs}) == 8
        assert {(c["wqe_builder"], c["wqe_batch"]) for c in cs} == {
            (builder, batch) for builder in ("legacy", "direct") for batch in (32, 64, 128, 256)
        }
        assert all(c["self_copy_mode"] == "self_copy_after_submit" and c["visit"] == 32 for c in cs)


def test_ub_layout_budget():
    for capacity in (8, 64):
        for qps in (1, 2):
            for batch in (32, 64, 128, 256):
                headers = batch * (64 + 32)
                sender = headers + capacity * qps * (128 + 8)
                end = sender + capacity * 256 * 4
                assert sender % 32 == 0 and end <= 224 * 1024 - (8 * 1024 + 64)


def test_comparison_uses_matched_round_baseline(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/clos"))
    import bench_wqe_sweep as experiment

    def measurement(builder, batch, bandwidth):
        return dict(
            config=dict(
                id=f"rank_p2_w1_q1_v32__self_copy_after_submit__{builder}_b{batch}",
                wqe_builder=builder,
                wqe_batch=batch,
            ),
            mean_us=1000 / bandwidth,
            GBps=bandwidth,
            samples_per_rank=30,
            per_sample_max_rank_us=dict(communication=1000 / bandwidth, notify=20, reduce=80),
        )

    legacy = [measurement("legacy", 32, 100), measurement("legacy", 32, 200)]
    direct = [measurement("direct", 128, 120), measurement("direct", 128, 220)]
    report = dict(
        results=[
            dict(
                shape=dict(tokens=4096, hidden=2816, topk=8, aiv=8),
                operations=dict(
                    combine_weighted=dict(
                        coarse=[legacy[0], direct[0]], finalists=[dict(repeats=legacy), dict(repeats=direct)]
                    )
                ),
            )
        ]
    )
    rows = experiment.comparison_rows(report)
    selected = [r for r in rows if r["builder"] == "direct"]
    assert [r["bandwidth_gain_percent"] for r in selected] == pytest.approx([20, 20, 10])
    assert all(r["notify_max_rank_us"] == 20 and r["reduce_max_rank_us"] == 80 for r in rows)
