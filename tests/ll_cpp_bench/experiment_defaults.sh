#!/usr/bin/env bash
# Shared defaults for the two convenience entry points. Source from Bash.
ll_bench_profiles=ref-profile,remote-profile,pipeline-profile,return-profile
ll_bench_cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.T101}
ll_bench_mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
ll_bench_shmem=${SHMEM_ROOT:-$root/third-party/shmem/install/shmem}

ll_bench_environment() {
    local selected_cann=$1 selected_mpi=$2 selected_shmem=$3
    set --
    # CANN's environment script may read optional, unset variables.
    if [[ -f "$selected_cann/set_env.sh" ]]; then
        set +u
        source "$selected_cann/set_env.sh"
        set -u
    fi
    export ASCEND_HOME_PATH="$selected_cann" MPI_HOME="$selected_mpi" SHMEM_ROOT="$selected_shmem"
    export PATH="$MPI_HOME/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
    export LD_LIBRARY_PATH="$MPI_HOME/lib:$MPI_HOME/lib64:$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
}
