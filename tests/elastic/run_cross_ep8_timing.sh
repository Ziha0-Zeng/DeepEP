#!/usr/bin/env bash
# Machine-specific reproduction: 90.90.93.84 + 90.90.93.184, NPUs 12..15 each.
set -eo pipefail
node_rank=${1:?Usage: bash run_cross_ep8_timing.sh NODE_RANK SHARED_RUN_NAME [extra benchmark options]}
run_name=${2:?Use the same unique run name on both machines}
shift 2
[[ "$node_rank" == 0 || "$node_rank" == 1 ]] || { echo 'NODE_RANK must be 0 or 1' >&2; exit 2; }
source /home/z00964904/zj1031/ascend-toolkit/set_env.sh
set -u
export SHMEM_ROOT=/home/z00906330/DeepEP/third-party/shmem/install/shmem
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export DEEPEP_LL_RANKS_PER_HOST=4
export GLOO_SOCKET_IFNAME=enp101s0f0
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
unset DEEPEP_LL_INIT_TRACE
ulimit -Sn 65536
cd /home/z00825995/DeepEP
exec timeout -k 15s 300s /root/miniforge3/envs/py311_tyf/bin/python -u tests/elastic/test_ep_low_latency.py \
  --num-processes 4 --num-nodes 2 --node-rank "$node_rank" \
  --master-addr 90.90.93.84 --master-port "${DEEPEP_TEST_MASTER_PORT:-30011}" \
  --shmem-ip-port "${DEEPEP_TEST_SHMEM_ENDPOINT:-tcp://30.30.30.129:30012}" \
  --backend rdma --group-backend gloo --num-cpu-timeout-secs 180 \
  --num-tokens 32 48 64 72 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype bf16 \
  --do-cpu-sync 0 --route normal --seed 0 \
  --warmup 10 --num-iters 50 --perf-tail 50 --perf-timer device-barrier \
  --repeat-combine 2 --correctness-repeats 3 \
  --output-dir /home/z00825995/DeepEP/results/ll-combine-cross-ep8 \
  --run-name "$run_name" "$@"
