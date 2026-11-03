"""CPU-only regression checks for cross-host benchmark timing boundaries."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("ll_timer_st", Path(__file__).with_name("test_ep_low_latency.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

@pytest.mark.parametrize("backend", ["gloo", "hccl"])
def test_hccl_gate_stream_dependency(monkeypatch, backend):
    calls = []
    world, group, token = object(), object(), object()
    def reduce(value, **kwargs):
        assert value is token
        assert kwargs == dict(op="MAX", group=world if backend == "hccl" else group, async_op=True)
        calls.append("reduce")
        return SimpleNamespace(wait=lambda: calls.append("wait"))
    def new_group(**kwargs):
        assert kwargs["backend"] == "hccl"
        calls.append("new_group")
        return group
    fake_dist = SimpleNamespace(get_backend=lambda: backend, group=SimpleNamespace(WORLD=world),
                                new_group=new_group, all_reduce=reduce,
                                ReduceOp=SimpleNamespace(MAX="MAX"), get_rank=lambda: 0)
    fake_torch = SimpleNamespace(distributed=fake_dist, float32="float32",
                                 ones=lambda *a, **kw: token,
                                 npu=SimpleNamespace(synchronize=lambda: calls.append("synchronize"),
                                    current_stream=lambda: stream))
    class Stream:
        npu_stream = property(lambda self: calls.append("flush") or 123)
    stream = Stream()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    monkeypatch.setattr("time.sleep", lambda duration: calls.append(("delay", duration)))
    gate = st.HcclBenchmarkGate("npu:0", 120, delay_ms=50)
    assert calls[-4:] == ["reduce", "wait", "flush", "synchronize"]
    calls.clear()
    gate()
    assert calls == [("delay", .05), "reduce", "wait", "flush"]

@pytest.mark.parametrize("argv", [
    ["--benchmark-gate-delay-ms", "-1"],
    ["--benchmark-gate-delay-ms", "nan"],
    ["--benchmark-gate-delay-ms", "50", "--backend", "mte"],
    ["--verify-device-timer", "--perf-timer", "event"],
])
def test_invalid_timer_validation_options(argv):
    with pytest.raises(ValueError):
        st.validate_args(st.make_parser().parse_args(argv), 8)

def test_native_gate_after_validation_before_events():
    source = (Path(__file__).resolve().parents[2] / "csrc/ops/low_latency.cpp").read_text()
    body = source[source.index('module.def("combine_low_latency"'):]
    assert body.index('args.tilingData = get(') < body.index('benchmark_gate();')
    assert body.index('benchmark_gate();') < body.index('timer.begin(stream);')
    assert body.index('timer.begin(stream);') < body.index('combine_server_dedup_kernel_do(stream, &args);')
    assert body.index('combine_server_dedup_kernel_do(stream, &args);') < body.index('timer.end(stream);')
    assert "Cross-host timing requires an HCCL device gate" in body
    assert body.index('launch_gate.arm(stream);') < body.index('benchmark_gate();')
    assert body.index('timer.end(stream);') < body.index('launch_gate.release();')
