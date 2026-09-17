#!/usr/bin/env bash
# From the PC: start the four ranks as resident servers. Rank 0 (the first
# address) answers the OpenAI-compatible API on HTTP_PORT; the others wait
# for its requests. Ranks are detached on the boards and survive this shell.
#
#   board/tp/serve-tp.sh 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
#   board/tp/serve-tp.sh stop 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
#
# Then: curl http://10.42.0.137:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
set -uo pipefail
if [ "${1:-}" = stop ]; then shift; for ip in "$@"; do ssh -n -o BatchMode=yes ubuntu@$ip 'pkill tp_llama' 2>/dev/null; echo "$ip: stopped"; done; exit 0; fi
ips=("$@"); nranks=${#ips[@]}
peers=$(IFS=,; echo "${ips[*]}")
model=${MODEL:-'~/llama-3.2-1b-instruct-q4_0-pure.gguf'}
port=${HTTP_PORT:-8080}
for r in $(seq 0 $((nranks - 1))); do
    timeout 30 ssh -n -o BatchMode=yes ubuntu@${ips[$r]} "pkill tp_llama; cd ~/board/tp && setsid nohup ./tp_llama --model $model --rank $r --nranks $nranks --peers $peers --serve $port > ~/tp-serve.log 2>&1 < /dev/null &" && echo "${ips[$r]}: rank $r started"
done
echo "waiting for the API..."
for i in $(seq 1 60); do
    if curl -s -m 2 "http://${ips[0]}:$port/health" >/dev/null 2>&1; then curl -s "http://${ips[0]}:$port/health"; echo; exit 0; fi
    sleep 1
done
echo "API did not come up; see ~/tp-serve.log on the boards" >&2; exit 1
