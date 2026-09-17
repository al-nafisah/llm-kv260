#!/usr/bin/env bash
# From the PC: start one tp_llama rank on each board, in parallel, and
# collect the output. Rank order = argument order; rank 0 prints the text.
#
#   board/tp/run-tp.sh "prompt" 64 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
#   board/tp/run-tp.sh "prompt" 64 10.42.0.137                  # single rank, no network
set -uo pipefail
prompt=${1:?prompt}; n=${2:?tokens}; shift 2
ips=("$@"); nranks=${#ips[@]}
peers=$(IFS=,; echo "${ips[*]}")
model=${MODEL:-'~/llama-3.2-1b-q4_0-pure.gguf'}
out=${OUT:-/tmp/tp-run}; mkdir -p "$out"
# STAGGER=s delays rank r by r*s seconds (to test late peers); TP_DEBUG=1 traces the exchanges;
# BIN=./tp_llama.old runs another binary in ~/board/tp (an A/B against a kept build).
for r in $(seq 0 $((nranks - 1))); do
    (sleep $(( r * ${STAGGER:-0} )); timeout ${LIMIT:-600} ssh -n -o BatchMode=yes ubuntu@${ips[$r]} "cd ~/board/tp && TP_DEBUG=${TP_DEBUG:-} ${BIN:-./tp_llama} --model $model --rank $r --nranks $nranks --peers $peers --n $n --prompt \"$prompt\"" > "$out/rank$r.out" 2> "$out/rank$r.err"; ssh -n -o BatchMode=yes ubuntu@${ips[$r]} 'pkill tp_llama' 2>/dev/null) &
done
wait
echo "=== rank 0 text"; cat "$out/rank0.out"
echo "=== per rank"; for r in $(seq 0 $((nranks - 1))); do grep -h "generated" "$out/rank$r.err" || tail -2 "$out/rank$r.err"; done
