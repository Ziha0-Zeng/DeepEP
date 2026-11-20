# Combine WQE experiment

Rebuild and install **both** the native extension and device kernel library from
this branch before testing. Production defaults remain `legacy` / 32. This is an
ordinary non-expanded V2 **peer combine** experiment; dispatch, token scheduling,
early-single and aggregate pipeline are not modified. Unsupported opt-in paths
fail on the host instead of silently using another builder.

```bash
# Inspect the complete plan without torch/NPU or SHMEM initialization.
python benchmarks/clos/bench_wqe_sweep.py --plan-only

# Single server, continuous EP8. Eight communication AIVs, 32 reduce AIVs.
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_wqe_sweep.py \
  --output /tmp/ep8_wqe_run1
```

Defaults: T4096/H2816/K8/E256/BF16/expand0, G8,
rank/P2/W1/Q1/visit32, self-copy after submit, 1 GiB pages enabled,
L2 eviction disabled. These are explicit matched experiment settings, not the
automatic-policy baseline. Change page/L2 settings explicitly if the environment
requires it (`--no-huge-pages`, `--flush-l2-cache`).

The complete experiment uses **one Buffer/SHMEM initialization per rank**. It
reuses routing and inputs, measures cached dispatch as an unchanged reference,
and tests unweighted/weighted combine with both builders at 32/64/128/256.
Use `--strategies rank_p1_w1_q1_v32 rank_p2_w1_q1_v32` to compare peer ownership;
use exact IDs supported by `bench_fast_sweep.py` for Q2 or other visit values.
Route generation and the independent reduce core count remain fixed per case.

Each configuration gets 5 warmups and 10 coarse samples. The top two and all
matched legacy32 baselines are rerun in forward/reverse order with 30 samples
each. Use `--warmup 10` for ten warmups in all phases. No numerical precision
check is performed. Candidate activation drains previous work before changing
host options; all ranks must use the same settings. Before each invocation L2
eviction, if enabled, completes before the device barrier on the existing Buffer
stream. That barrier is excluded from measured stages.

Controls are read on every combine call, without changing Buffer allocation:

```bash
export DEEPEP_CLOS_COMBINE_WQE_BUILDER=direct # legacy / direct
export DEEPEP_CLOS_COMBINE_WQE_BATCH=128      # 32 / 64 / 128 / 256
```

The sweep sets these itself and verifies the loaded native extension recognizes
them. Clear them before launching the sweep. In `direct`, a single span maps
threads directly to row/field; mixed spans use an ordered-prefix binary search.
Indices remain in MTE-filled UB windows. Payload still uses UDMA. Each batch
uses the configured number of SIMT threads; threads beyond the actual count exit.

The batch setting is a **limit**, not an observed batch fill. Peer visit budgets,
QP rotation, queue-state changes, index-window refill flushes and completion
markers retain their existing semantics. Increasing the limit may therefore
have little effect, particularly with Q2 or many short visits. The experiment
does not yet instrument actual WQE fill/flush counts or SQ waiting time.

UB ownership: the same AIV owns its SQ state, WQE scratch and peer index windows.
Pending spans must be flushed before an index window is overwritten. WQE scratch
is reused only after SHMEM's MTE3 read completion. Sender scratch is
`batch * 96 + capacity * QPs * 136` bytes; index windows add
`capacity * 256 * 4` bytes. The largest specialization (256, capacity64, Q2)
uses 107520 bytes, below the kernel UB budget. Payload, weight destinations,
peer/row ownership, SO publication, quiet and final barriers are unchanged.

Artifacts: per-rank profiler exports, manifests and raw samples; `report.json`,
`summary.csv`, `best_configs.json`, and `wqe_comparison.csv/json`. Comparison rows
contain separate communication/notify/reduce durations and bandwidth improvement
against the matched legacy32 reference in the same round. Main timing is the
mean of per-sample rank maxima. Bandwidth uses mean per-rank deduplicated BF16
payload bytes including self, excludes FP32 weights, and uses decimal GB/s.
Notify/reduce maxima are independent and must not be added to claim API timing.

CPU tests cover host option validation, one-initialization orchestration and the
actual WQE builder body against an independent address oracle, including mixed
spans, tails, fields and invalid indices. They do not qualify device ordering or
NPU performance. Run device precision/stress regression before adopting a winner.
