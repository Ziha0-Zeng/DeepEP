#!/usr/bin/env bash
# Build the frozen original and the isolated diagnostic experiments.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
variants="base,$ll_bench_profiles"
cann=$ll_bench_cann mpi=$ll_bench_mpi shmem=$ll_bench_shmem jobs=4 dry=0
fail() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
build_all.sh [--variants NAME[,NAME...]] [--jobs N] [--dry-run]
             [--cann-home DIR] [--mpi-home DIR] [--shmem-root DIR]
With no arguments, build base and the four current diagnostic experiments:
ref-profile, remote-profile, pipeline-profile, return-profile.
Previous experiments opt-profile, r36-profile and early-profile remain
available through --variants but are excluded from the default build.
ops-shmem-base builds the optional ops-transformer baseline with SHMEM transport.
Existing build directories are updated in place; all flags are explicit.
SDK defaults: CANN cann-9.2.T101, MPI /usr/local/mpich-3.2.1,
             SHMEM third-party/shmem/install/shmem in this checkout.
ASCEND_HOME_PATH, MPI_HOME and SHMEM_ROOT override these SDK defaults.
USAGE
            exit 0;;
        --dry-run) dry=1; shift;;
        --variants|--jobs|--cann-home|--mpi-home|--shmem-root)
            (($#>=2)) || fail "Missing value: $1"
            key=$1 value=$2; shift 2
            case "$key" in
                --variants) variants=$value;; --jobs) jobs=$value;; --cann-home) cann=$value;;
                --mpi-home) mpi=$value;; --shmem-root) shmem=$value;;
            esac;;
        *) fail "Unknown build option: $1";;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] && ((jobs<=64)) || fail 'jobs must be in [1,64]'
[[ $variants =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || fail 'Provide comma-separated experiment names'
IFS=, read -r -a names <<< "$variants"
# Validate the complete selection before starting any build.
for name in "${names[@]}"; do
    case "$name" in
        base|ops-shmem-base|ref-profile|opt-profile|r36-profile|early-profile|remote-profile|pipeline-profile|return-profile) ;;
        *) fail "Unknown experiment: $name (see --help)";;
    esac
done
if ((!dry)); then ll_bench_environment "$cann" "$mpi" "$shmem"; fi
for name in "${names[@]}"; do
    variant=opt profile=ON small=OFF early=OFF remote=OFF defer=OFF recv=0
    case "$name" in
        base) variant=base; profile=OFF;;
        ops-shmem-base) variant=ops-shmem-base; profile=OFF;;
        opt-profile) small=ON;;
        r36-profile) recv=36;;
        early-profile) early=ON;;
        remote-profile) remote=ON;;
        pipeline-profile) defer=ON;;
        return-profile) remote=ON; defer=ON;;
    esac
    cmd=(bash "$root/tests/ll_cpp_bench/build.sh"
         --cann-home "$cann" --mpi-home "$mpi" --shmem-root "$shmem"
         --build-dir "$root/build/ll-cpp-$name" --jobs "$jobs"
         --variant "$variant" --profile "$profile" --small-batch "$small"
         --recv-cores "$recv" --early-reduce "$early" --remote-first "$remote" --defer-return "$defer")
    printf '\nBuilding %s\n' "$name"
    if ((dry)); then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
printf '\nBuild selection complete: %s\n' "$variants"
