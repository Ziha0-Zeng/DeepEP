# Copyright (c) 2026, Lu Lu
"""A5 HT gate orchestration; run this file from the trusted CI checkout."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def configuration(env):
    devices = env.get("DEEPEP_CI_DEVICES", "").split(",")
    if len(devices) not in (2, 4, 8) or any(not re.fullmatch(r"[0-9]+", d) for d in devices):
        raise ValueError("DEEPEP_CI_DEVICES must list exactly 2, 4 or 8 reserved device IDs")
    if len(set(map(int, devices))) != len(devices):
        raise ValueError("Device IDs must be distinct")
    ports = [int(env[name]) for name in ("DEEPEP_CI_MASTER_PORT", "DEEPEP_CI_SHMEM_PORT")]
    if any(not 1024 <= p <= 65535 for p in ports) or len(set(ports)) != 2:
        raise ValueError("Use two different reserved ports in 1024..65535")
    if env.get("CI_MODE") not in ("quick", "full"):
        raise ValueError("CI_MODE must be quick or full")
    return len(devices), ports


def cases(world, mode):
    if world not in (2, 4, 8) or mode not in ("quick", "full"):
        raise ValueError("Unsupported test matrix")
    tokens = 33 if mode == "quick" else 257
    common = [
        "--ep-size",
        str(world),
        "--nproc-per-node",
        str(world),
        "--tokens",
        str(tokens),
        "--topk",
        "6",
        "--experts",
        "256",
        "--route",
        "all",
        "--repeat",
        "3",
    ]
    result = []
    for dedup in ["server", "rank"] if world == 8 else ["server"]:
        flags = [*common, f"--{dedup}-dedup"]
        result.append((f"dispatch-{dedup}", "test_dispatch.py", [*flags, "--dtype", "both"]))
        for dtype in ("bf16", "fp8"):
            result.append(
                (
                    f"combine-{dedup}-{dtype}",
                    "test_combine.py",
                    [*flags, "--with-dispatch", "--with-weights", "--dispatch-dtype", dtype, "--no-use-1g-huge-pages"],
                )
            )
        result.append(
            (
                f"combine-{dedup}-no-weights",
                "test_combine.py",
                [*flags, "--with-dispatch", "--without-weights", "--no-use-1g-huge-pages"],
            )
        )
    return result


def verify_ranks(directory, world, source_hash, combine=False):
    paths = sorted(directory.glob("rank-*.json"))
    if {p.name for p in paths} != {f"rank-{r}.json" for r in range(world)}:
        raise ValueError("Missing or unexpected rank reports")
    for rank in range(world):
        data = json.loads((directory / f"rank-{rank}.json").read_text())
        if (
            type(data.get("rank")) is not int
            or data["rank"] != rank
            or type(data.get("world_size")) is not int
            or data["world_size"] != world
            or data.get("passed") is not True
            or type(data.get("checks")) is not int
            or data["checks"] < 1
            or data.get("source_sha256") != source_hash
        ):
            raise ValueError(f"Invalid rank report: {rank}")
        if combine and (data.get("dispatch_checked") is not True or data.get("notify_checked") is not True):
            raise ValueError("Combine did not validate native Dispatch and Notify")


def stop_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            continue
        # The launcher may exit before its workers; still kill remaining group members.
        if sig == signal.SIGTERM:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return


def descendants(parents, root):
    found = set()
    pending = {root}
    while pending:
        children = {pid for pid, parent in parents.items() if parent in pending} - found
        found.update(children)
        pending = children
    return found


def stop_children():
    # Package validation creates new sessions for build/test commands. A process
    # group alone cannot contain those. As a Linux subreaper, we also own any
    # workers orphaned when a launcher exits and can clean them before unlocking.
    for _ in range(3):
        parents = {}
        for path in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
                parents[int(path.parent.name)] = int(fields["PPid"])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        owned = descendants(parents, os.getpid())
        if not owned:
            return
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break


def run(command, *, cwd, env, log, timeout):
    print("Running:", " ".join(map(str, command)), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=timeout)
            if status:
                raise subprocess.CalledProcessError(status, command)
        finally:
            stop_group(process)
            stop_children()


def interrupted(signum, _frame):
    raise RuntimeError(f"CI interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if sys.platform != "linux" or not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        parser.error("Run on Linux with Python assertions enabled")
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Keep CI artifacts outside the source checkout")
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child process cleanup")
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTHONPATH", None)
    report = {
        "schema": 1,
        "status": "failure",
        "stages": [],
        **{
            key: env.get(name, "")
            for key, name in (
                ("head_sha", "CI_HEAD_SHA"),
                ("base_sha", "CI_BASE_SHA"),
                ("trusted_sha", "CI_TRUSTED_SHA"),
                ("run_id", "GITHUB_RUN_ID"),
                ("run_attempt", "GITHUB_RUN_ATTEMPT"),
                ("mode", "CI_MODE"),
                ("pr", "CI_PR"),
            )
        },
    }

    def stage(name, action):
        entry = {"name": name, "status": "failure"}
        report["stages"].append(entry)
        started = time.monotonic()
        try:
            action()
            entry["status"] = "success"
        finally:
            entry["seconds"] = round(time.monotonic() - started, 3)

    try:
        world, (master_port, shmem_port) = configuration(env)
        report["world"] = world
        env["ASCEND_RT_VISIBLE_DEVICES"] = env["DEEPEP_CI_DEVICES"]
        env["DEEPEP_SHMEM_ENDPOINT"] = f"tcp://127.0.0.1:{shmem_port}"

        def check_environment():
            actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if actual != env["CI_HEAD_SHA"]:
                raise ValueError("Checkout does not match the requested PR revision")
            ancestor = subprocess.run(
                ["git", "-C", str(source), "merge-base", "--is-ancestor", env["CI_BASE_SHA"], actual],
                check=False,
            )
            if ancestor.returncode != 0:
                raise ValueError("PR head must contain the current base commit. Update the PR and rerun A5 CI.")
            run(
                [sys.executable, source / "scripts/record_shmem_sdk.py", "--sdk", env["SHMEM_ROOT"], "--check"],
                cwd=output,
                env=env,
                log=output / "environment.log",
                timeout=120,
            )

        stage("environment", check_environment)
        package = output / "package"
        command = [
            sys.executable,
            source / "scripts/validate_package.py",
            "--output",
            package,
            "--system-site-packages",
            "--runtime-ranks",
            str(world),
        ]
        if env.get("DEEPEP_CI_WHEELHOUSE"):
            command += ["--wheelhouse", env["DEEPEP_CI_WHEELHOUSE"]]

        def check_package():
            run(command, cwd=output, env=env, log=output / "package.log", timeout=5400)
            validation = json.loads((package / "validation.json").read_text())
            wheels = list((package / "dist").glob("*.whl"))
            if len(wheels) != 1 or hashlib.sha256(wheels[0].read_bytes()).hexdigest() != validation["wheel_sha256"]:
                raise ValueError("Wheel identity mismatch")
            if validation["runtime_ranks"] != world:
                raise ValueError("Runtime probe used a different world size")
            report["wheel_sha256"] = validation["wheel_sha256"]
            report["source_sha256"] = validation["build_info"]["source_sha256"]

        stage("package", check_package)
        for name, script, flags in cases(world, env["CI_MODE"]):
            directory = output / "ht" / name

            def check(script=script, flags=flags, directory=directory, name=name):
                run(
                    [
                        package / "venv/bin/python",
                        package / "tests/elastic" / script,
                        *flags,
                        "--master-port",
                        str(master_port),
                        "--output",
                        directory,
                    ],
                    cwd=package,
                    env=env,
                    log=output / f"{name}.log",
                    timeout=1200,
                )
                verify_ranks(directory, world, report["source_sha256"], combine=script == "test_combine.py")

            stage(name, check)
        report["status"] = "success"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "result.json")


if __name__ == "__main__":
    main()
