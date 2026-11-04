#!/usr/bin/env python3
"""Summarise a session.py directory: rate, time split and module power per phase.

    python3 board/tp/bench/session-report.py OUT > report.txt

Power is the mean of the INA260 samples (PC clock) inside a window:
- API replies: first to last streamed token (decode), request to first token (prefill).
- run-tp.sh: run-tp only reports when every rank has exited, so the decode
  window is [end - gen_s + 1.5 s, end - 2.5 s], which lies inside generation
  for any teardown between 0 and 2.5 s.
- idle and bench phases: the whole phase; llama-bench: the phase minus its first 30 s (load).
"""
import glob, json, os, re, statistics as st, sys

d = sys.argv[1]
power = {}
for f in sorted(glob.glob(f"{d}/power-*.tsv")):
    host = os.path.basename(f)[6:-4]
    power[host] = [tuple(map(float, l.split())) for l in open(f) if l.strip()]
hosts = sorted(power)
phases = {}
for l in open(f"{d}/phases.tsv"):
    n, a, b = l.split("\t"); phases[n] = (float(a), float(b))

def win(host, a, b):
    v = [w for t, w in power[host] if a <= t <= b]
    return (st.mean(v), min(v), max(v), len(v)) if v else (float("nan"),) * 3 + (0,)

def prow(a, b, only=None):
    cells, tot = [], 0.0
    for h in hosts:
        m, lo, hi, n = win(h, a, b)
        if only is None or h in only: tot += m
        cells.append(f"{m:.3f} [{lo:.2f}-{hi:.2f}, n={n}]")
    return cells, tot

print(f"# session {d}")
print("# power: K26 module INA260 (hwmon power1_input), W; mean [min-max, samples] per board; total = sum of the boards' means")
print("# boards:", " ".join(hosts))
print()
print("## idle")
for n in ("api-idle", "idle-no-runtime", "idle-after"):
    if n in phases:
        c, t = prow(*phases[n]); print(f"{n:16s} total {t:.3f} W | " + " | ".join(c))
print()
print("## API replies (temperature 0, streamed, from the PC over the LAN)")
for f in sorted(glob.glob(f"{d}/api-*.json")):
    n = os.path.basename(f)[:-5]; r = json.load(open(f)); tm = r["timings"]
    k = tm["predicted_n"]; dec = r["t_last"] - r["t_first"]
    c, t = prow(r["t_first"], r["t_last"]); cp, tp_ = prow(r["t_req"], r["t_first"])
    lc = r["stats"].get("last_chat", {}) if isinstance(r["stats"], dict) else {}
    print(f"{n}: prompt {tm['prompt_n']} tok, {k} tokens, server {tm['predicted_per_second']:.2f} t/s, "
          f"client {(k - 1) / dec:.2f} t/s (tokens 2..{k} over {dec:.2f} s), first token {r['t_first'] - r['t_req']:.3f} s; "
          f"decode power total {t:.3f} W, {t / tm['predicted_per_second']:.4f} J/token; prefill power total {tp_:.3f} W")
    print("    decode per board: " + " | ".join(c))
    print("    last_chat: " + json.dumps(lc, sort_keys=True))
print()
print("## run-tp.sh, 400 tokens after the 24-token prompt, greedy")
for tag in ("tp4", "tp2", "tp1"):
    for i in (1, 2, 3):
        n = f"{tag}-{i}"
        if n not in phases: continue
        txt = open(f"{d}/{n}.txt").read()
        m = re.search(r"rank 0: prompt (\d+) tokens in ([\d.]+) s.*?generated (\d+) tokens in ([\d.]+) s = ([\d.]+) t/s; per token: (.*)", txt)
        if not m: print(n, "no rank 0 line"); continue
        gen = float(m.group(4)); a, b = phases[n]
        used = hosts[: int(tag[2:])]
        c, t = prow(b - gen + 1.5, b - 2.5, used)
        print(f"{n}: {m.group(3)} tokens in {gen:.2f} s = {m.group(5)} t/s; per token {m.group(6)}; "
              f"decode power of the {len(used)} running boards {t:.3f} W, {t / float(m.group(5)):.4f} J/token")
        print("    per board: " + " | ".join(c))
print()
print("## bench_gemv2 and llama-bench")
for n, (a, b) in phases.items():
    if n.startswith("bench-") or n == "cpu-llama-bench":
        c, t = prow(a + (30 if n == "cpu-llama-bench" else 0), b)
        print(f"{n}: {b - a:.0f} s; power per board: " + " | ".join(c))
