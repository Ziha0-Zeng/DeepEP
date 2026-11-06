"""Host-only checks; actual MPI/SHMEM/NPU execution is a separate validation."""
import os
import hashlib
import json
import io
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

HERE = Path(__file__).resolve().parent


def test_original_kernel_snapshot_is_verbatim(tmp_path):
    cmake = shutil.which("cmake")
    git = shutil.which("git")
    if not cmake or not git:
        pytest.skip("CMake and Git are required")
    root = HERE.parents[1]
    revision = "5b38bf2ea1f67f38c0c2d3662618ae1a0457bfc6"
    wrapper = tmp_path / "snapshot.cmake"
    wrapper.write_text(f'set(DEEPEP_SOURCE_DIR "{root.as_posix()}")\n'
                       f'include("{(HERE / "baseline.cmake").as_posix()}")\n')
    subprocess.run([cmake, "-P", str(wrapper)], cwd=tmp_path, check=True, timeout=30)
    expected = subprocess.check_output([git, "-C", str(root), "archive", "--format=tar", revision, "csrc/kernels"])
    extracted = tmp_path / ("original-" + revision)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(expected)) as archive:
        for member in archive:
            if member.isfile():
                assert (extracted / member.name).read_bytes() == archive.extractfile(member).read()
                count += 1
    assert count > 100
    original = (extracted / "csrc/kernels/combine/low_latency.cpp").read_text(encoding="utf-8")
    assert "DEEPEP_LL_COMBINE_SMALL_BATCH" not in original
    assert "CombineProfile" not in original
    assert "ll_bench_entry_gate" not in original


def test_options_routes_and_statistics(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for host C++ checks")
    executable = tmp_path / "options.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", str(HERE / "options_test.cpp"),
                    "-o", str(executable)], check=True, timeout=180)
    subprocess.run([str(executable)], check=True, timeout=30)


@pytest.fixture
def bash():
    candidate = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not candidate or not Path(candidate).exists():
        pytest.skip("Bash unavailable")
    return candidate


def test_shell_syntax(bash):
    for script in ("build.sh", "run.sh", "launch.sh", "prepare_shmem.sh", "hydra_proxy_command.sh",
                   "build_all.sh", "run_all.sh", "experiment_defaults.sh"):
        subprocess.run([bash, "-n", str(HERE / script)], check=True, timeout=10)


@pytest.mark.parametrize("soft,hard,expected,ok", [
    ("1024", "1048576", "65536", True),
    ("131072", "1048576", "131072", True),
    ("unlimited", "unlimited", "unlimited", True),
    ("1024", "unlimited", "65536", True),
    ("1024", "4096", None, False),
])
def test_per_node_nofile_setup(bash, soft, hard, expected, ok):
    text = (HERE / "launch.sh").read_text(encoding="utf-8")
    begin = text.index("prepare_nofile() {")
    function = text[begin:text.index("\n}\n", begin) + 3]
    # Exercise the actual launcher function with a resource-limit shell stub.
    script = r'''
set -eu
OSTYPE=linux-gnu
node_rank=1
mock_soft=$1 mock_hard=$2
fail() { echo "ERROR: $*" >&2; exit 2; }
ulimit() {
    if [[ $1 == -Hn ]]; then printf '%s\n' "$mock_hard"; return; fi
    [[ $1 == -Sn ]]
    if (($#==2)); then mock_soft=$2; else printf '%s\n' "$mock_soft"; fi
}
''' + function + '\nprepare_nofile\nprintf "effective=%s hard=%s\\n" "$mock_soft" "$mock_hard"\n'
    result = subprocess.run([bash, "-c", script, "_", soft, hard],
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    if ok:
        assert f"effective={expected} hard={hard}" in result.stdout
        assert f"node=1 nofile soft={expected} hard={hard}" in result.stdout
    else:
        assert "raise this shell/container hard limit" in result.stderr
        assert "effective=" not in result.stdout


@pytest.mark.parametrize("nested", [False, True])
def test_sdk_manifest_record_and_check(tmp_path, nested):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake unavailable")
    sdk = tmp_path / "sdk"
    (sdk / "include").mkdir(parents=True)
    (sdk / "lib").mkdir()
    header = b"// fixture\n"
    (sdk / "include/test.h").write_bytes(header)
    (sdk / "lib/libshmem.so").write_bytes(b"sdk fixture")
    (sdk / "lib/aclshmem_bootstrap_mpi.so").write_bytes(b"mpi fixture")
    (tmp_path / "dependencies.lock.json").write_text(json.dumps({"shmem": {
        "revision": "fixture", "header_sha256_lf": {"test.h": hashlib.sha256(header).hexdigest()}}}))
    build = tmp_path / "build"
    flags = build / ("src/device/" if nested else "") / "CMakeFiles/aclshmem_device.dir/flags.make"
    flags.parent.mkdir(parents=True)
    flags.write_text("CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=1\n")
    cann = tmp_path / "cann"
    (cann / "compiler").mkdir(parents=True)
    (cann / "compiler/version.info").write_text("Version=9.3.0\ntimestamp=fixture\n")
    common = [cmake, f"-DPROJECT_ROOT={tmp_path}", f"-DSDK={sdk}", f"-DSDK_BUILD={build}",
              f"-DASCEND_HOME_PATH={cann}"]
    def run(mode, success=True):
        result = subprocess.run([*common, f"-DMODE={mode}", "-P", str(HERE / "sdk_identity.cmake")],
                                capture_output=True, text=True, timeout=10)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
    run("record")
    identity = json.loads((sdk / "deepep-sdk.json").read_text())
    assert identity["build"]["hns_1825_pi_on_chip"] == 1
    assert identity["build"]["rdma_backend"] == "HNS_1825"
    assert len(identity["sha256"]) == 3
    run("check")
    run("record", success=False)
    (sdk / "lib/aclshmem_bootstrap_mpi.so").write_bytes(b"changed")
    run("check", success=False)


def test_mpi_launch_commands(bash):
    for hosts, launcher, ranks in (("localhost", "fork", 16), ("node0,node1", "manual", 32)):
        result = subprocess.run([bash, str(HERE / "run.sh"), "--hosts", hosts,
                                 "--launcher", launcher, "--master-addr", "node0", "--dry-run", "--",
                                 "--tokens", "72", "--output-dir", "/tmp/test data"],
                                capture_output=True, text=True, timeout=10, check=True)
        assert f"-ppn 16 -n {ranks}" in result.stdout
        if launcher == "manual":
            assert "-launcher ssh -launcher-exec" in result.stdout
            assert "hydra_proxy_command.sh" in result.stdout
            assert "-launcher manual" not in result.stdout
        else:
            assert f"-launcher {launcher}" in result.stdout
        assert "--tokens 72" in result.stdout
        assert "MPIEXEC_PORT_RANGE=29671:29771" in result.stdout
    for ep in (2,4,8,16):
        result = subprocess.run([bash, str(HERE / "run.sh"), "--hosts", "localhost", "--ep", str(ep),
                                 "--launcher", "fork", "--dry-run"], capture_output=True, text=True, timeout=10, check=True)
        assert f"-ppn {ep} -n {ep}" in result.stdout
    for args in (("--hosts", "a,a"), ("--hosts", "a,b", "--launcher", "fork"),
                 ("--port-range", "3000:2000"), ("--hosts", "a,b,c"), ("--hosts", "a,b", "--ep", "8"),
                 ("--hosts", "a", "--ep", "32")):
        result = subprocess.run([bash, str(HERE / "run.sh"), *args, "--dry-run"],
                                capture_output=True, timeout=10)
        assert result.returncode != 0


def test_per_node_launch_arguments(bash):
    for ep in (2, 4, 8, 16):
        result = subprocess.run([bash, str(HERE / "launch.sh"), "--ep", str(ep), "--variants", "base,opt",
                                 "--dry-run"], capture_output=True, text=True, check=True, timeout=10)
        assert result.stdout.count(f"-ppn {ep} -n {ep}") == 1
        assert "-launcher fork" in result.stdout
        assert "0-base" in result.stdout and "1-opt" in result.stdout
    for node in (0, 1):
        result = subprocess.run([bash, str(HERE / "launch.sh"), "--ep", "32", "--hosts", "node0,node1",
                                 "--node-rank", str(node), "--master-port", "31000", "--variants", "base,opt",
                                 "--run-id", "trial", "--dry-run", "--", "--tokens", "72"],
                                capture_output=True, text=True, check=True, timeout=10)
        assert f"--proxy-id {node}" in result.stdout
        assert "--control-port node0:31000" in result.stdout
        assert "--control-port node0:31001" not in result.stdout
        if node == 0:
            assert "--variants base,opt" in result.stdout.replace("\\,", ",")
        assert ("-n 32" in result.stdout) == (node == 0)
        if node == 0:
            assert "HYDRA_LAUNCHER_AUTOFORK=0" in result.stdout
            assert "MPIEXEC_PORT_RANGE=31000:31000" in result.stdout
            assert "-launcher ssh -launcher-exec" in result.stdout
        assert "--launcher manual" not in result.stdout
    for args in (("--ep", "4", "--node-rank", "1"), ("--ep", "32"), ("--run-id", "../escape"),
                 ("--master-port", "65536", "--variants", "base,opt"), ("--variants", "../escape"),
                 ("--variants", "base,,opt")):
        result = subprocess.run([bash, str(HERE / "launch.sh"), *args, "--dry-run"],
                                capture_output=True, timeout=10)
        assert result.returncode != 0


def test_repeated_single_host_profile_comparison(bash, tmp_path):
    """Run the same command twice with fake binaries; preserve both sets of results."""
    root = tmp_path / "project"
    script = root / "tests/ll_cpp_bench/launch.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "launch.sh", script)
    mpi = tmp_path / "mpi"
    helpers = tmp_path / "helpers"

    def write(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, newline="\n")
        path.chmod(0o755)

    write(helpers / "setsid", 'exec "$@"\n')
    write(helpers / "timeout", 'exit 1\n')  # Only availability is checked for one host.
    write(mpi / "bin/mpiexec", r'''
while (($#)); do
    if [[ $1 == */ll_combine_bench ]]; then exec "$@"; fi
    shift
done
exit 1
''')
    for variant in ("ref-profile", "opt-profile"):
        write(root / f"build/ll-cpp-{variant}/ll_combine_bench", r'''
while (($#)); do
    case $1 in --output-dir) root_out=$2;; --variants) variants=$2;; --profile) level=$2;; --entry-sync) sync=$2;; esac
    shift 2
done
[[ $level == 2 && $sync == device ]]
echo init >>"$root_out/initializations"
IFS=, read -r -a phases <<< "$variants"
for phase in "${!phases[@]}"; do
    out="$root_out/$phase-${phases[$phase]}"
    mkdir -p "$out"
    for ((r=0;r<4;r++)); do echo passed >"$out/rank-$r.PASS"; done
    echo "ll-cpp-${phases[$phase]}" >"$out/summary.csv"
    echo profile-fixture >"$out/rank-0-profile.csv"
done
''')
    for variant in ("ref-profile", "opt-profile"):
        (root / f"build/ll-cpp-{variant}/libll_bench_variant.so").touch()
    write(script.with_name("summarize.sh"), 'echo "profile-summary: $1"\n')
    helper_path = subprocess.check_output([bash, "-c", 'cygpath -u "$1"' if os.name == "nt" else 'printf "%s" "$1"',
                                          "_", helpers.as_posix()], text=True).strip()
    original_path = subprocess.check_output([bash, "-c", 'printf "%s" "$PATH"'], text=True)
    env = dict(os.environ, PATH=helper_path + ":" + original_path)
    args = [bash, str(script), "--ep", "4", "--variants", "ref-profile,opt-profile",
            "--mpi-home", mpi.as_posix(), "--run-id", "repeat", "--",
            "--profile", "2", "--entry-sync", "device"]
    for _ in range(2):
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count("profile-summary:") == 2
    runs = list((root / "results/ll-cpp/repeat").iterdir())
    assert len(runs) == 2
    for run in runs:
        assert (run / "initializations").read_text().splitlines() == ["init"]
        assert (run / "node-0/suite.log").exists()
        for phase, variant in enumerate(("ref-profile", "opt-profile")):
            output = run / f"{phase}-{variant}"
            assert len(list(output.glob("rank-*.PASS"))) == 4
            assert f"ll-cpp-{variant}" in (output / "summary.csv").read_text()
            assert (output / "rank-0-profile.csv").read_text().strip() == "profile-fixture"
            assert (output / "profile-summary-node-0.txt").exists()


def test_hydra_command_launcher_only_prints(bash, tmp_path):
    proxy = tmp_path / "path with spaces" / "hydra_pmi_proxy"
    proxy.parent.mkdir()
    marker = tmp_path / "unexpected-execution"
    proxy.write_text(f"#!/usr/bin/env bash\ntouch '{marker.as_posix()}'\n", newline="\n")
    proxy.chmod(0o755)
    result = subprocess.run([bash, str(HERE / "hydra_proxy_command.sh"), "-x", "node0",
                             '"' + proxy.as_posix() + '"', "--control-port", "node0:29671",
                             "--launcher", "ssh", "--usize", "-2", "--proxy-id", "0"],
                            capture_output=True, text=True, check=True, timeout=10)
    assert "LL_HYDRA_PROXY_READY 0 node0:29671" in result.stdout
    assert "env -u HYDI_CONTROL_FD" in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("failure", [0, 1, 2])
def test_two_node_script_sequence_with_fake_hydra(bash, tmp_path, failure):
    """Exercise two independent scripts. This fixture is not an MPI runtime test."""
    root = tmp_path / "project"
    script = root / "tests/ll_cpp_bench/launch.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "launch.sh", script)
    command_launcher = script.with_name("hydra_proxy_command.sh")
    shutil.copyfile(HERE / "hydra_proxy_command.sh", command_launcher)
    command_launcher.chmod(0o755)
    mpi = tmp_path / "mpi"
    (mpi / "bin").mkdir(parents=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    def write(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, newline="\n")
        path.chmod(0o755)
    for variant in ("base", "opt"):
        write(root / f"build/ll-cpp-{variant}/ll_combine_bench", "exit 0\n")
        (root / f"build/ll-cpp-{variant}/libll_bench_variant.so").touch()
    # Git Bash lacks setsid on some Windows installations. Process-group and
    # signal behavior belongs to Linux validation; this test checks orchestration.
    helpers = tmp_path / "helpers"
    write(helpers / "setsid", 'exec "$@"\n')
    write(helpers / "timeout", 'shift 3\nexec "$@"\n')
    write(mpi / "bin/mpiexec", r'''
if [[ ${1:-} == -version ]]; then echo '    Version: 3.2.1'; exit 0; fi
[[ $HYDRA_LAUNCHER_AUTOFORK == 0 ]]
port=${MPIEXEC_PORT_RANGE%:*}
while (($#)); do
    case $1 in -launcher) launcher=$2;; -launcher-exec) launcher_exec=$2;; --variants) variants=$2;; esac
    if [[ $1 == --output-dir ]]; then out=$2; break; fi
    shift
done
[[ $launcher == ssh && $launcher_exec == */hydra_proxy_command.sh ]]
mkdir -p "$out"
touch "$FAKE_SHARED/controller-$port"
for node in 0 1; do
    bash "$launcher_exec" -x "node$node" '"/fixture/hydra_pmi_proxy"' --control-port "node0:$port" --proxy-id "$node"
done
if [[ ${FAKE_FAIL:-0} == 2 ]]; then echo 'fixture controller abort'; exit 134; fi
for ((i=0;i<200;i++)); do
    if [[ -f "$FAKE_SHARED/join-$port-0" && -f "$FAKE_SHARED/join-$port-1" ]]; then break; fi
    sleep 0.02
done
[[ -f "$FAKE_SHARED/join-$port-0" && -f "$FAKE_SHARED/join-$port-1" ]]
if [[ ${FAKE_FAIL:-0} == 1 ]]; then touch "$FAKE_SHARED/done-$port"; exit 17; fi
IFS=, read -r -a phases <<< "$variants"
for phase in "${!phases[@]}"; do
    phase_out="$out/$phase-${phases[$phase]}"
    mkdir -p "$phase_out"
    for ((r=0;r<32;r++)); do echo passed >"$phase_out/rank-$r.PASS"; done
    echo summary >"$phase_out/summary.csv"
done
touch "$FAKE_SHARED/done-$port"
''')
    write(mpi / "bin/hydra_pmi_proxy", r'''
while (($#)); do
    case $1 in --control-port) port=${2##*:};; --proxy-id) node=$2;; esac
    shift 2
done
for ((i=0;i<200;i++)); do
    [[ -f "$FAKE_SHARED/controller-$port" ]] && break
    sleep 0.02
done
[[ -f "$FAKE_SHARED/controller-$port" ]]
if [[ ${FAKE_FAIL:-0} == 2 ]]; then echo 'fixture connection refused'; exit 1; fi
touch "$FAKE_SHARED/join-$port-$node"
for ((i=0;i<200;i++)); do
    [[ -f "$FAKE_SHARED/done-$port" ]] && exit 0
    sleep 0.02
done
exit 1
''')
    env = dict(os.environ, FAKE_SHARED=shared.as_posix(), FAKE_FAIL=str(failure))
    # Git Bash accepts Windows drive paths as arguments, but PATH uses /c/... .
    helper_path = subprocess.check_output([bash, "-c", 'cygpath -u "$1"' if os.name == "nt" else 'printf "%s" "$1"',
                                          "_", helpers.as_posix()], text=True).strip()
    original_path = subprocess.check_output([bash, "-c", 'printf "%s" "$PATH"'], text=True)
    env["PATH"] = helper_path + ":" + original_path
    args = [bash, str(script), "--ep", "32", "--hosts", "node0,node1", "--mpi-home", mpi.as_posix(),
            "--variants", "base,opt", "--run-id", "test", "--join-timeout", "5", "--job-timeout", "10"]
    processes = [subprocess.Popen([*args, "--node-rank", str(node)], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) for node in (1, 0)]
    try:
        for node, proc in zip((1, 0), processes):
            output, _ = proc.communicate(timeout=25)
            assert (proc.returncode == 0) == (failure == 0), output
            if node == 0 and failure == 2:
                assert "fixture controller abort" in output
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
    assert (shared / "controller-29671").exists()
    assert not (shared / "controller-29672").exists()
