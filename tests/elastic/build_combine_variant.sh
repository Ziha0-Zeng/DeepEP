#!/usr/bin/env bash
# Rebuild a complete, explicit A/B variant; run the same command on both hosts.
set -eo pipefail
variant=${1:?Usage: build_combine_variant.sh base|r36|small|early|remote|return [OFF|ON]}
profile=${2:-OFF}
case "$profile" in ON|OFF) ;; *) echo 'Profile must be ON or OFF' >&2; exit 2;; esac
export DEEPEP_LL_COMBINE_PROFILE=$profile
export DEEPEP_LL_COMBINE_SMALL_BATCH=OFF DEEPEP_LL_COMBINE_RECV_CORES=0
export DEEPEP_LL_COMBINE_EARLY_REDUCE=OFF DEEPEP_LL_COMBINE_REMOTE_FIRST=OFF DEEPEP_LL_COMBINE_DEFER_RETURN=OFF
case "$variant" in
    base) ;;
    r36) export DEEPEP_LL_COMBINE_RECV_CORES=36;;
    small) export DEEPEP_LL_COMBINE_SMALL_BATCH=ON;;
    early) export DEEPEP_LL_COMBINE_EARLY_REDUCE=ON;;
    remote) export DEEPEP_LL_COMBINE_REMOTE_FIRST=ON;;
    return) export DEEPEP_LL_COMBINE_REMOTE_FIRST=ON DEEPEP_LL_COMBINE_DEFER_RETURN=ON;;
    *) echo "Unknown variant: $variant" >&2; exit 2;;
esac
source /home/z00964904/zj1031/ascend-toolkit/set_env.sh
set -u
export DEEPEP_NPU_ARCH=Ascend950 CMAKE_BUILD_PARALLEL_LEVEL=4
export SHMEM_ROOT=/home/z00906330/DeepEP/third-party/shmem/install/shmem
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
cd /home/z00825995/DeepEP
python=/root/miniforge3/envs/py311_tyf/bin/python
"$python" setup.py build_ext --inplace
# Source timestamps on these hosts can predate the packaging cache.
"$python" setup.py build_py --force
"$python" -m pip install --no-build-isolation --no-deps --force-reinstall .
printf 'Installed variant=%s profile=%s\n' "$variant" "$profile"
