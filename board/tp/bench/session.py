#!/usr/bin/env python3
"""One recorded measurement session (buildlog 0013), run from the PC.

Samples every K26 module's INA260 (hwmon power1_input) about 10 times a
second for the whole session, timestamped on the PC clock, and runs in
order: the API idle, 400-token and short API replies, the boards idle with
no runtime, run-tp.sh on 4, 2 and 1 boards, bench_gemv2 on each board, and
llama.cpp's llama-bench on board 1's CPUs. Then it restarts the API.

    python3 board/tp/bench/session.py OUT 10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114

OUT gets power-<host>.tsv (pc_time watts), phases.tsv (name start end),
one file per run, and env.txt. ONLY=cpu runs the CPU baseline alone. board/tp/bench/session-report.py reads it.
"""
import json, os, subprocess, sys, threading, time, urllib.request

out, ips = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
tp = os.path.dirname(here)
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
API = f"http://{ips[0]}:8080"
LONG = ("Write a detailed history of the Roman Empire, from its founding to its fall, "
        "covering the key events and figures.")
SHORT = "Explain in a few sentences why the sky is blue."
MODEL = "~/llama-3.2-1b-instruct-q4_0-pure.gguf"
phases = open(f"{out}/phases.tsv", "a", buffering=1)

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def sh(cmd, path=None, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if path:
        with open(path, "w") as f:
            f.write(f"### {' '.join(cmd)}\n### exit {r.returncode}\n{r.stdout}{r.stderr}")
    return r

def remote(ip, cmd, path=None, timeout=900):
    return sh(SSH + [f"ubuntu@{ip}", cmd], path, timeout)

class phase:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time(); log("begin", self.name); return self
    def __exit__(self, *exc):
        phases.write(f"{self.name}\t{self.t0:.3f}\t{time.time():.3f}\n"); log("end", self.name)

# The sampler reads the sysfs file with the shell's read builtin on CPU 1
# (the runtime is on CPU 2, the NIC's interrupt on CPU 3).
SAMPLER = ("h=$(grep -l ina260 /sys/class/hwmon/hwmon*/name); p=${h%/name}/power1_input; "
           "exec taskset -c 1 bash -c 'while :; do read -r w < '$p'; echo $w; sleep 0.1; done'")
procs = []
def sampler(ip):
    host = remote(ip, "hostname").stdout.strip()
    p = subprocess.Popen(SSH + [f"ubuntu@{ip}", SAMPLER], stdout=subprocess.PIPE, text=True)
    procs.append(p)
    with open(f"{out}/power-{host}.tsv", "a", buffering=1) as f:
        for line in p.stdout:
            f.write(f"{time.time():.3f}\t{int(line) / 1e6:.4f}\n")

def env():
    cmd = ("hostname; uname -r; uptime; ip -o link show eth0 | grep -o 'mtu [0-9]*'; "
           "md5sum ~/gemv2/*.bit.bin; ls /sys/class/uio/*/name | xargs cat; "
           "irq() { grep -i display /proc/interrupts | awk '{s+=$2+$3+$4+$5} END {print s+0}'; }; "
           "a=$(irq); sleep 2; echo display irqs/2s $(( $(irq) - a )); "
           "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | sort | uniq -c; free -m | head -2; "
           "pgrep -a tp_llama || echo 'no tp_llama'; date -u +'board clock %FT%TZ'")
    with open(f"{out}/env.txt", "a") as f:
        f.write(f"### PC {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        for ip in ips:
            f.write(f"===== {ip}\n{remote(ip, cmd).stdout}")

def chat(prompt, n, path):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"{API}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "llm-kv260-session/1.0"})
    t_req = time.time(); t_first = None; text = []; last = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            chunk = json.loads(line[6:]); last = chunk
            d = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if d:
                if t_first is None: t_first = time.time()
                text.append(d); t_last = time.time()
    stats = json.load(urllib.request.urlopen(f"{API}/stats", timeout=10))
    rec = {"t_req": t_req, "t_first": t_first, "t_last": t_last, "timings": last.get("timings"),
           "usage": last.get("usage"), "stats": stats, "text": "".join(text)}
    json.dump(rec, open(path, "w"), indent=1)
    tm = rec["timings"] or {}
    log(f"  {tm.get('predicted_n')} tokens, server {tm.get('predicted_per_second', 0):.2f} t/s, "
        f"first token {t_first - t_req:.2f} s")
    return rec

def run_tp(tag, k, boards):
    for i in range(1, k + 1):
        with phase(f"{tag}-{i}"):
            e = dict(os.environ, MODEL=MODEL, OUT=f"{out}/{tag}-{i}")
            r = subprocess.run(["bash", f"{tp}/run-tp.sh", LONG, "400", *boards],
                               capture_output=True, text=True, env=e, timeout=900)
            open(f"{out}/{tag}-{i}.txt", "w").write(r.stdout + r.stderr)
            log("  " + (r.stdout.strip().splitlines() or ["?"])[-len(boards)])
        time.sleep(10)

def cpu_baseline():
    with phase("cpu-llama-bench"):
        remote(ips[0], f"hostname; ~/llama.cpp/build/bin/llama-bench -m {MODEL} -p 0 -n 400 -r 3 -t 4 -ngl 0 -dev none",
               f"{out}/cpu-llama-bench.txt", timeout=3600)

for ip in ips:
    threading.Thread(target=sampler, args=(ip,), daemon=True).start()
time.sleep(5)
if os.environ.get("ONLY") == "cpu":        # the CPU baseline alone, API stopped around it
    sh(["bash", f"{tp}/serve-tp.sh", "stop", *ips], f"{out}/serve-stop-cpu.txt")
    time.sleep(10)
    with phase("idle-no-runtime-cpu"): time.sleep(30)
    cpu_baseline()
    time.sleep(10)
    sh(["bash", f"{tp}/serve-tp.sh", *ips], f"{out}/serve-start-cpu.txt")
    for p in procs: p.terminate()
    log("done"); sys.exit(0)
env()
with phase("api-idle"): time.sleep(60)
for i in range(1, 4):
    with phase(f"api-400-{i}"): chat(LONG, 400, f"{out}/api-400-{i}.json")
    time.sleep(10)
for i in range(1, 4):
    with phase(f"api-short-{i}"): chat(SHORT, 128, f"{out}/api-short-{i}.json")
    time.sleep(5)
sh(["bash", f"{tp}/serve-tp.sh", "stop", *ips], f"{out}/serve-stop.txt")
time.sleep(10)
with phase("idle-no-runtime"): time.sleep(60)
run_tp("tp4", 3, ips)
run_tp("tp2", 3, ips[:2])
run_tp("tp1", 3, ips[:1])
VEC = "k256_n32 k512_n64 k1024_n32 k2048_n64 k8192_n32 k512_n2048 k2048_n768 k2048_n2048 k2048_n4096"
for ip in ips:
    with phase(f"bench-{ip}"):
        remote(ip, f"hostname; cd ~/kv260gemv && taskset -c 2 ./bench_gemv2 ~/vectors udmabuf0 {VEC}",
               f"{out}/bench-{ip}.txt")
    time.sleep(5)
cpu_baseline()
time.sleep(10)
with phase("idle-after"): time.sleep(30)
env()
sh(["bash", f"{tp}/serve-tp.sh", *ips], f"{out}/serve-start.txt")
for p in procs: p.terminate()
log("done")
