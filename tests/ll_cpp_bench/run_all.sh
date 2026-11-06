#!/usr/bin/env bash
# One command per host; node 1 only needs the fresh run-id printed by node 0.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
ep=32 node_rank=${NODE_RANK:-} hosts="${HOST0:-90.90.93.84},${HOST1:-90.90.93.184}"
master= port=29671 variants=$ll_bench_profiles run_id= dry=0
cann=$ll_bench_cann mpi=$ll_bench_mpi shmem=$ll_bench_shmem
job_timeout=3600 join_timeout=300
bench_keys=(tokens hidden topk experts capacity dtype route seed entry-sync warmup iters profile timeout)
declare -A bench=([tokens]=32,48,64,72,256 [hidden]=3584 [topk]=8 [experts]=384
                  [capacity]=256 [dtype]=bf16 [route]=random [seed]=0 [entry-sync]=auto
                  [warmup]=10 [iters]=10 [profile]=2 [timeout]=300)
fail() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
Node 0: run_all.sh
Node 1: run_all.sh RUN_ID
Options: --ep 2|4|8|16|32 --node-rank 0|1 --hosts HOST0,HOST1
         --master-addr IP --master-port PORT --run-id ID --variants NAME[,NAME...]
         --cann-home DIR --mpi-home DIR --shmem-root DIR --dry-run
         --join-timeout SECONDS --job-timeout SECONDS
Benchmark options (with or without a preceding --):
         --tokens LIST --devices LIST --hidden N --topk N --experts N --capacity N
         --dtype bf16|fp16 --route random|hotspot --seed N --entry-sync auto|host|device
         --warmup N --iters N --profile 0|1|2 --timeout SECONDS
Defaults: EP32, hosts 90.90.93.84,90.90.93.184, port 29671.
          Four diagnostic versions; BS 32,48,64,72,256; H3584/K8/E384/BF16.
          Warmup 10, samples 10, profile 2; Host MPI sync for EP32.
All selected variants share one MPI/ACL/SHMEM initialization.
Single-host EP2/4/8/16 defaults to device entry sync.
Node is detected from local addresses; --node-rank overrides detection.
Node 0 generates a new run-id on every invocation unless --run-id is supplied.
For custom hosts, ports or variants, use the printed node-1 command.
Benchmark shapes and sampling options are propagated by MPI from node 0.
USAGE
            exit 0;;
        --dry-run) dry=1; shift;;
        --) shift;;
        --ep|--node-rank|--hosts|--master-addr|--master-port|--variants|--run-id|--mpi-home|--cann-home|--shmem-root|--job-timeout|--join-timeout)
            (($#>=2)) || fail "Missing value: $1"
            key=$1 value=$2; shift 2
            case "$key" in
                --ep) ep=$value;; --node-rank) node_rank=$value;; --hosts) hosts=$value;;
                --master-addr) master=$value;; --master-port) port=$value;; --variants) variants=$value;;
                --run-id) run_id=$value;; --mpi-home) mpi=$value;; --cann-home) cann=$value;;
                --shmem-root) shmem=$value;; --job-timeout) job_timeout=$value;; --join-timeout) join_timeout=$value;;
            esac;;
        --tokens|--devices|--hidden|--topk|--experts|--capacity|--dtype|--route|--seed|--entry-sync|--warmup|--iters|--profile|--timeout)
            (($#>=2)) || fail "Missing value: $1"
            bench[${1#--}]=$2; shift 2;;
        -*) fail "Unknown option: $1";;
        *) [[ -z $run_id ]] || fail 'Specify only one run-id'; run_id=$1; shift;;
    esac
done
[[ $ep =~ ^(2|4|8|16|32)$ ]] || fail 'EP must be 2,4,8,16 or 32'
if ((ep==32)); then
    [[ $hosts =~ ^[a-zA-Z0-9_.-]+,[a-zA-Z0-9_.-]+$ ]] || fail 'EP32 requires two hosts'
    IFS=, read -r -a nodes <<< "$hosts"
    [[ ${nodes[0]} != "${nodes[1]}" ]] || fail 'EP32 requires two distinct hosts'
    master=${master:-${nodes[0]}}
    if [[ -z $node_rank ]]; then
        local_ids=$(hostname -I 2>/dev/null || true)
        local_ids+=" $(hostname 2>/dev/null || true) $(hostname -f 2>/dev/null || true)"
        if command -v ip >/dev/null; then
            local_ids+=" $(ip -o -4 addr show 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true)"
        fi
        for index in 0 1; do
            for address in $local_ids; do
                if [[ $address == "${nodes[$index]}" ]]; then
                    [[ -z $node_rank || $node_rank == "$index" ]] || fail 'Both hosts match local addresses; specify --node-rank'
                    node_rank=$index
                fi
            done
        done
        [[ -n $node_rank ]] || fail "Cannot identify this host in $hosts; set --node-rank 0 or 1"
    fi
else
    node_rank=${node_rank:-0}
    hosts=localhost
fi
[[ $node_rank =~ ^[01]$ ]] || fail 'node-rank must be 0 or 1'
[[ $ep == 32 || $node_rank == 0 ]] || fail 'Single-host EP uses node-rank 0'
if [[ -z $run_id ]]; then
    [[ $node_rank == 0 ]] || fail 'Copy RUN_ID from node 0: bash tests/ll_cpp_bench/run_all.sh RUN_ID'
    run_id="ep${ep}-all-$(date +%Y%m%d-%H%M%S)-$$-$RANDOM"
fi
[[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail 'Use letters, digits, _, . or - for run-id'
[[ ${bench[profile]} =~ ^[012]$ ]] || fail 'profile must be 0, 1 or 2'
if [[ ( ,$variants, == *,base,* || ,$variants, == *,ops-shmem-base,* ) && ${bench[profile]} != 0 ]]; then
    fail 'base and ops-shmem-base are uninstrumented; use --profile 0, or ref-profile for diagnostics'
fi
printf 'RUN_ID=%s\nnode=%s EP=%s variants=%s\n' "$run_id" "$node_rank" "$ep" "$variants"
if [[ $node_rank == 0 ]]; then
    printf 'BS=%s warmup=%s iters=%s profile=%s\n' "${bench[tokens]}" "${bench[warmup]}" "${bench[iters]}" "${bench[profile]}"
else
    echo 'Benchmark shapes and sampling options come from node 0.'
fi
if [[ $ep == 32 && $node_rank == 0 ]]; then
    peer_cmd=(bash tests/ll_cpp_bench/run_all.sh "$run_id")
    # With defaults this prints only RUN_ID; custom launch settings travel too.
    if [[ $hosts != 90.90.93.84,90.90.93.184 || $master != 90.90.93.84 || $port != 29671 || $variants != "$ll_bench_profiles" ]]; then
        peer_cmd+=(--node-rank 1 --hosts "$hosts" --master-addr "$master" --master-port "$port" --variants "$variants")
    fi
    if [[ ${bench[profile]} != 2 ]]; then peer_cmd+=(--profile "${bench[profile]}"); fi
    if [[ $job_timeout != 3600 || $join_timeout != 300 ]]; then
        peer_cmd+=(--job-timeout "$job_timeout" --join-timeout "$join_timeout")
    fi
    if [[ $cann != "$ll_bench_cann" ]]; then peer_cmd+=(--cann-home "$cann"); fi
    if [[ $mpi != "$ll_bench_mpi" ]]; then peer_cmd+=(--mpi-home "$mpi"); fi
    if [[ $shmem != "$ll_bench_shmem" ]]; then peer_cmd+=(--shmem-root "$shmem"); fi
    printf 'Run on node 1: '; printf '%q ' "${peer_cmd[@]}"; printf '\n'
fi
if ((!dry)); then ll_bench_environment "$cann" "$mpi" "$shmem"; fi
args=(--ep "$ep" --node-rank "$node_rank" --hosts "$hosts" --master-port "$port"
      --variants "$variants" --run-id "$run_id" --mpi-home "$mpi"
      --job-timeout "$job_timeout" --join-timeout "$join_timeout")
if [[ -n $master ]]; then args+=(--master-addr "$master"); fi
if ((dry)); then args+=(--dry-run); fi
args+=(--)
for key in "${bench_keys[@]}"; do args+=("--$key" "${bench[$key]}"); done
if [[ ${bench[devices]+set} ]]; then args+=(--devices "${bench[devices]}"); fi
exec bash "$root/tests/ll_cpp_bench/launch.sh" "${args[@]}"
