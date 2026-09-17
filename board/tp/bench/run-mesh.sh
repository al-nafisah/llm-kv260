#!/usr/bin/env bash
# From the PC: run mesh_bench on every board given (rank order = argument order) and print each rank's report.
#   bash board/tp/bench/run-mesh.sh 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
set -uo pipefail
ips=("$@"); nranks=${#ips[@]}
peers=$(IFS=,; echo "${ips[*]}")
out=${OUT:-/tmp/mesh-run}; mkdir -p "$out"
for r in $(seq 0 $((nranks - 1))); do
    timeout ${LIMIT:-300} ssh -n -o BatchMode=yes ubuntu@${ips[$r]} "cd ~/board/tp/bench && ${PREFIX:-} ./mesh_bench --rank $r --nranks $nranks --peers $peers --iters ${ITERS:-1000} ${MESH_ARGS:-}" > "$out/rank$r.out" 2> "$out/rank$r.err" &
done
wait
for r in $(seq 0 $((nranks - 1))); do echo "=== rank $r (${ips[$r]})"; cat "$out/rank$r.out"; tail -2 "$out/rank$r.err"; done
