#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
set -euo pipefail

# Provisioned by the runner administrator, never taken from the PR checkout.
config=/etc/deepep-ci/runner.env
if [[ ! -f "$config" ]]; then
    echo "Missing $config; follow docs/ci.md" >&2
    exit 1
fi
source "$config"
: "${DEEPEP_CI_DEVICES:?Set the reserved device IDs in runner.env}"
: "${DEEPEP_CI_MASTER_PORT:?Set a reserved master port in runner.env}"
: "${DEEPEP_CI_SHMEM_PORT:?Set a reserved SHMEM port in runner.env}"
: "${SHMEM_ROOT:?Provide the pinned, registered SHMEM SDK}"
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh in runner.env}"
: "${DEEPEP_CI_LOCK:?Set one shared absolute machine lock path}"
export DEEPEP_CI_DEVICES DEEPEP_CI_MASTER_PORT DEEPEP_CI_SHMEM_PORT SHMEM_ROOT ASCEND_HOME_PATH
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE=OFF DEEPEP_BUILD_EP_GMM_FUSED=OFF
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH PYTHONOPTIMIZE
command -v flock >/dev/null
command -v g++ >/dev/null

# All DeepEP runners on this host must use the same lock. The administrator
# reserves the corresponding devices and ports from non-CI workloads too.
exec 9>"$DEEPEP_CI_LOCK"
flock --exclusive --timeout 600 9
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${DEEPEP_CI_PYTHON:-python3}" "$script_dir/run_a5.py" --source "$1" --output "$2"
