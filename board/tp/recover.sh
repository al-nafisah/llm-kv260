#!/usr/bin/env bash
# From the PC, after a power cut: bring the four-board stream back without
# touching a board by hand. Waits for the boards to boot, loads the gemv2
# overlay and the memory pools where they are missing, restarts the four
# ranks, then checks the LAN API with a real chat.
#
#   bash board/tp/recover.sh                                   # the four boards, rank order
#   bash board/tp/recover.sh 10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114
#
# Needs passwordless sudo for load-overlay.sh on every board
# (/etc/sudoers.d/load-overlay), and the PC's key on each. PUBLIC is the
# API to check, board 1's port 8080 on the LAN unless overridden; HTTP_PORT
# and MODEL pass through to serve-tp.sh.
# "No route to host" on all four right after power-on means they are still
# booting (about a minute), not the UFW trap in docs/kv260-bringup.md.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ips=("$@"); [ ${#ips[@]} -gt 0 ] || ips=(10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114)
public=${PUBLIC:-http://${ips[0]}:${HTTP_PORT:-8080}}
ua='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/154 Safari/537.36'   # Cloudflare 403s Python-urllib
ssh_q() { ssh -n -o ConnectTimeout=3 -o BatchMode=yes "ubuntu@$1" "$2" 2>/dev/null; }

up=0
for i in $(seq 1 48); do
    up=0; for ip in "${ips[@]}"; do ssh_q "$ip" true && up=$((up + 1)); done
    [ "$up" -eq ${#ips[@]} ] && break
    sleep 5
done
[ "$up" -eq ${#ips[@]} ] || { echo "only $up of ${#ips[@]} boards answer ssh after 4 min" >&2; exit 1; }

for ip in "${ips[@]}"; do
    printf '%s ' "$ip"
    ssh_q "$ip" 'if [ "$(cat /sys/class/uio/*/name | grep -c gemv)" != 2 ]; then sudo -n bash ~/board/load-overlay.sh ~/gemv2 >/dev/null 2>&1; fi
        echo "$(hostname): up $(uptime -p | sed "s/^up //") · gemv uio $(cat /sys/class/uio/*/name | grep -c gemv) · pools $(ls /dev/udmabuf? 2>/dev/null | wc -l) · mtu $(cat /sys/class/net/eth0/mtu)"'
    n=$(ssh_q "$ip" 'cat /sys/class/uio/*/name | grep -c gemv; ls /dev/udmabuf? 2>/dev/null | wc -l' | paste -sd' ')
    [ "$n" = "2 3" ] || { echo "$ip: overlay or pools missing ($n); load-overlay.sh needs the sudoers rule" >&2; exit 1; }
done

bash "$here/serve-tp.sh" "${ips[@]}" | tail -1 || exit 1

for i in $(seq 1 36); do
    h=$(curl -s -A "$ua" --max-time 5 "$public/health" || true)
    case "$h" in *'"ok"'*) echo "LAN API after ~$((i * 5)) s: $h"; break;; esac
    sleep 5
done
case "${h:-}" in *'"ok"'*) ;; *) echo "$public/health never answered ok; check ~/tp-serve.log on board 1" >&2; exit 1;; esac

curl -s -A "$ua" --max-time 90 "$public/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Explain what an FPGA is to a high-school student."}],"max_tokens":200}' \
| python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("chat: %d tokens at %.1f tokens/s, first token after %.2f s" % (t["predicted_n"], t["predicted_per_second"], t["prompt_ms"]/1000))'
