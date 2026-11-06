#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
cann=${ASCEND_HOME_PATH:-}
shmem=${SHMEM_ROOT:-}
build="$root/build/ll-cpp-base"
small=OFF early=OFF profile=OFF remote_first=OFF defer_return=OFF recv=0 jobs=4 variant=auto ops_source=${OPS_SOURCE_DIR:-}
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'build.sh --shmem-root SDK [--cann-home SDK] [--mpi-home DIR] [--build-dir DIR]'
            echo '         [--variant base|opt|ops-shmem-base|auto] [--profile ON|OFF] [--small-batch ON|OFF] [--early-reduce ON|OFF] [--recv-cores 0|8..36] [--jobs N]'
            echo '         [--ops-source DIR] optional pinned ops-transformer checkout'
            echo '         [--remote-first ON|OFF] [--defer-return ON|OFF]'
            exit 0;;
        --shmem-root|--cann-home|--mpi-home|--build-dir|--variant|--profile|--small-batch|--early-reduce|--recv-cores|--jobs|--remote-first|--defer-return|--ops-source)
            (($# >= 2)) || { echo "Missing value: $1" >&2; exit 2; }
            key=$1 value=$2; shift 2
            case "$key" in
                --shmem-root) shmem=$value;; --cann-home) cann=$value;; --mpi-home) mpi=$value;;
                --build-dir) build=$value;; --small-batch) small=$value;; --early-reduce) early=$value;;
                --variant) variant=$value;; --profile) profile=$value;; --recv-cores) recv=$value;; --jobs) jobs=$value;;
                --remote-first) remote_first=$value;; --defer-return) defer_return=$value;;
                --ops-source) ops_source=$value;;
            esac;;
        *) echo "Unknown build option: $1" >&2; exit 2;;
    esac
done
[[ $profile =~ ^(ON|OFF)$ && $small =~ ^(ON|OFF)$ && $early =~ ^(ON|OFF)$ && $recv =~ ^(0|[89]|[12][0-9]|3[0-6])$ ]] || { echo 'Invalid Combine build option' >&2; exit 2; }
[[ $variant =~ ^(base|opt|ops-shmem-base|auto)$ ]] || { echo 'Unknown variant' >&2; exit 2; }
[[ $remote_first =~ ^(ON|OFF)$ && $defer_return =~ ^(ON|OFF)$ ]] || { echo 'Return options must be ON or OFF' >&2; exit 2; }
if [[ $variant == auto ]]; then
    variant=base
    if [[ $profile == ON || $small == ON || $early == ON || $recv != 0 || $remote_first == ON || $defer_return == ON ]]; then variant=opt; fi
fi
original=OFF ops_base=OFF
if [[ $variant == ops-shmem-base ]]; then
    [[ $profile == OFF && $small == OFF && $early == OFF && $recv == 0 && $remote_first == OFF && $defer_return == OFF ]] || { echo 'ops-shmem-base is a clean independent kernel' >&2; exit 2; }
    ops_base=ON
    if [[ $build == "$root/build/ll-cpp-base" ]]; then build="$root/build/ll-cpp-ops-shmem-base"; fi
fi
if [[ $variant == base ]]; then
    [[ $profile == OFF && $small == OFF && $early == OFF && $recv == 0 && $remote_first == OFF && $defer_return == OFF ]] || { echo 'Use --variant opt for experimental options' >&2; exit 2; }
    original=ON
fi
[[ $jobs =~ ^[0-9]+$ ]] && ((jobs>=1 && jobs<=64)) || { echo 'jobs must be in [1,64]' >&2; exit 2; }
[[ -x "$mpi/bin/mpicxx" && -x "$cann/tools/bisheng_compiler/bin/bisheng" ]] || { echo 'Load CANN and select the MPI installation first' >&2; exit 2; }
[[ -f "$shmem/deepep-sdk.json" && -f "$shmem/lib/aclshmem_bootstrap_mpi.so" ]] || { echo 'Select the identified HNS_1825 SHMEM SDK with its MPI bootstrap plugin' >&2; exit 2; }
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$mpi/lib64:$shmem/lib:${LD_LIBRARY_PATH:-}"
cmake "-DPROJECT_ROOT=$root" "-DSDK=$shmem" -DMODE=check \
    -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
cmake -S "$root/tests/ll_cpp_bench" -B "$build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release "-DMPI_CXX_COMPILER=$mpi/bin/mpicxx" \
    "-DASCEND_HOME_PATH=$cann" "-DSHMEM_ROOT=$shmem" \
    "-DDEEPEP_LL_OPS_BASE=$ops_base" "-DLL_OPS_SOURCE_DIR=$ops_source" \
    "-DDEEPEP_LL_COMBINE_ORIGINAL=$original" "-DDEEPEP_LL_COMBINE_PROFILE=$profile" "-DDEEPEP_DEVICE_JOBS=$jobs" "-DDEEPEP_LL_COMBINE_SMALL_BATCH=$small" \
    "-DDEEPEP_LL_COMBINE_RECV_CORES=$recv" "-DDEEPEP_LL_COMBINE_EARLY_REDUCE=$early" \
    "-DDEEPEP_LL_COMBINE_REMOTE_FIRST=$remote_first" "-DDEEPEP_LL_COMBINE_DEFER_RETURN=$defer_return"
cmake --build "$build" --parallel "$jobs"
printf 'Built runner: %s/ll_combine_bench\nVariant module: %s/libll_bench_variant.so\n' "$build" "$build"
