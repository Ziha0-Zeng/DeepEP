"""Matched combine WQE builder/batch experiment, one Buffer per rank.

Defaults target single-server EP8, eight communication AIVs, H2816 BF16.
Uses bench_fast_sweep's capture, barriers and forward/reverse confirmation.
No numerical precision checks; candidate results are experimental only.
"""

import csv
import json
import os
from pathlib import Path
import sys

import bench_fast_sweep as sweep


def arguments(argv):
    supplied = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    defaults = {
        "--ep-size": ["8"],
        "--ranks-per-group": ["8"],
        "--case": ["4096,2816,8,8"],
        "--experts": ["256"],
        "--reduce-num-sms": ["32"],
        "--strategies": ["rank_p2_w1_q1_v32"],
        "--combine-self-copy-modes": ["self_copy_after_submit"],
        "--combine-wqe-batches": ["32", "64", "128", "256"],
        "--combine-wqe-builders": ["legacy", "direct"],
        "--warmup": ["5"],
        "--samples": ["10"],
        "--final-samples": ["30"],
        "--top": ["2"],
        "--output": ["wqe_sweep"],
        "--include-weighted-combine": [],
        "--payload-only-bandwidth": [],
        "--max-rank-timing": [],
    }
    return [part for key, values in defaults.items() if key not in supplied for part in [key, *values]] + argv


def comparison_rows(report):
    rows = []
    for case in report["results"]:
        for op, result in case["operations"].items():
            phases = {"coarse": result["coarse"]}
            for turn in (0, 1):
                phases[f"final{turn}"] = [c["repeats"][turn] for c in result["finalists"]]
            for phase, measurements in phases.items():
                baselines = {
                    r["config"]["id"].rsplit("__", 1)[0]: r
                    for r in measurements
                    if r["config"].get("wqe_builder") == "legacy" and r["config"].get("wqe_batch") == 32
                }
                for item in measurements:
                    config = item["config"]
                    baseline = baselines.get(config["id"].rsplit("__", 1)[0])
                    maxima = item["per_sample_max_rank_us"]
                    rows.append(
                        dict(
                            **case["shape"],
                            operator=op,
                            phase=phase,
                            config=config["id"],
                            builder=config.get("wqe_builder", "unchanged_dispatch"),
                            batch_limit=config.get("wqe_batch", 32),
                            comm_us=item["mean_us"],
                            GBps=item["GBps"],
                            notify_max_rank_us=maxima["notify"],
                            reduce_max_rank_us=maxima["reduce"],
                            bandwidth_gain_percent=(100 * (item["GBps"] / baseline["GBps"] - 1) if baseline else None),
                            samples_per_rank=item["samples_per_rank"],
                            precision_checked=False,
                        )
                    )
    return rows


def write_comparison(output):
    report = json.loads((output / "report.json").read_text(encoding="utf8"))
    rows = comparison_rows(report)
    sweep.write(output / "wqe_comparison.json", rows)
    with (output / "wqe_comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *arguments(sys.argv[1:])]
    args = sweep.parse_args()
    sweep.main()
    if not args.plan_only and int(os.environ["RANK"]) == 0:
        write_comparison(Path(args.output))
