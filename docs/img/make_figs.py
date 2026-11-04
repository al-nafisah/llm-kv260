#!/usr/bin/env python3
"""Charts for docs/img, parsed from the 0013 evidence files (no value is typed in here).

    python3 docs/img/make_figs.py

writes token.svg and scaling.svg, then a .pdf (for LaTeX) and a 2x .png of every
figure in this directory, the hand-written system.svg and engine.svg included.
Needs python3 (standard library only) and rsvg-convert (librsvg) for the copies.

Inputs, all under docs/buildlog/evidence/ (buildlog 0013, 2026-11-04):
  0013-tp-runs.txt          rank 0's summary line of each run-tp.sh run: prompt and
                            generated tokens, tokens/s, per-token engine / exchange / CPU
  0013-phases.txt           start and end of every phase of the session, PC clock
  0013-power-kv260-0N.txt   each K26 module's INA260, PC time and watts, ~10 samples/s
  0013-cpu-llama-bench.txt  llama-bench on kv260-01 without the engine
  0013-session-report.txt   the session's own summary, used here only as a cross-check

Module power uses the windows of board/tp/bench/session-report.py: a run-tp.sh
run is [end - generation time + 1.5 s, end - 2.5 s] summed over the boards that
ran; the llama-bench phase is its whole length minus the first 30 s, kv260-01
only. Energy per token is that power divided by the run's tokens/s. Every
computed power and energy is checked against 0013-session-report.txt, and the
conditions the captions state (prompt length, tokens generated, llama-bench
flags) are checked against the logs, so a caption cannot drift from its data.
"""
import datetime
import math
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "buildlog" / "evidence"
HOSTS = ["kv260-01", "kv260-02", "kv260-03", "kv260-04"]   # board 1..4 = rank 0..3

# ---------------------------------------------------------------- style
FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"
SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ENGINE, EXCHANGE, CPU = "#2a78d6", "#eb6834", "#1baf7a"    # the same three roles in every figure
ON_FILL = {ENGINE: "#ffffff", EXCHANGE: INK, CPU: INK}     # label colour inside each fill
BASELINE = "#52514e"                                         # CPU-only baseline marker

# Helvetica/Arial advance widths (1/1000 em), for placing labels without a renderer.
_ADV = dict(zip(" !\"#$%&'()*+,-./0123456789:;<=>?@", [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278] + [556] * 10 + [278, 278, 584, 584, 584, 556, 1015]))
_ADV.update(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611]))
_ADV.update(zip("abcdefghijklmnopqrstuvwxyz", [556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500]))


def tw(s, size, bold=False):
    """Approximate rendered width of s in px."""
    return sum(_ADV.get(c, 584) for c in s) * size / 1000 * (1.07 if bold else 1.0)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


HALO = f' stroke="{SURFACE}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'


def text(x, y, s, size=13, fill=INK, anchor="start", bold=False, extra=""):
    w = ' font-weight="bold"' if bold else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{a}{w}{extra}>{esc(s)}</text>'


def svg_doc(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-labelledby="title desc" font-family="{esc(FONT)}">\n'
            f'<title id="title">{esc(title)}</title>\n<desc id="desc">{esc(desc)}</desc>\n'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>\n' + "\n".join(body) + "\n</svg>\n")


def bar_end(x, y, w, h, r, fill, tip):
    """A bar segment whose right end is rounded (radius r) and left end square."""
    r = max(0.0, min(r, w, h / 2))
    d = (f"M{x:.2f},{y:.2f} H{x + w - r:.2f} A{r},{r} 0 0 1 {x + w:.2f},{y + r:.2f} "
         f"V{y + h - r:.2f} A{r},{r} 0 0 1 {x + w - r:.2f},{y + h:.2f} H{x:.2f} Z")
    return f'<path d="{d}" fill="{fill}"><title>{esc(tip)}</title></path>'


def legend(items, x, y, size=12.5):
    """items: (kind, colour, label); kind 'box' or 'dot' or 'diamond'. Returns svg pieces."""
    out = []
    for kind, colour, label in items:
        if kind == "box":
            out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="{colour}"/>')
        elif kind == "dot":
            out.append(f'<circle cx="{x + 6}" cy="{y - 4}" r="5" fill="{colour}"/>')
        else:
            out.append(diamond(x + 6, y - 4, 6, colour))
        out.append(text(x + 18, y, label, size, INK2))
        x += 18 + tw(label, size) + 26
    return out


def diamond(cx, cy, r, colour, tip=None):
    t = f"<title>{esc(tip)}</title>" if tip else ""
    return (f'<path d="M{cx:.2f},{cy - r:.2f} L{cx + r:.2f},{cy:.2f} L{cx:.2f},{cy + r:.2f} L{cx - r:.2f},{cy:.2f} Z" '
            f'fill="{SURFACE}" stroke="{colour}" stroke-width="2">{t}</path>')


# ---------------------------------------------------------------- evidence
def read_lines(name):
    return (EV / name).read_text().splitlines()


RUN_RE = re.compile(r"^rank 0: prompt (\d+) tokens in [\d.]+ s \([\d.]+ t/s\); generated (\d+) tokens in ([\d.]+) s = "
                    r"([\d.]+) t/s; per token: engine ([\d.]+) ms, comm ([\d.]+) ms, cpu ([\d.]+) ms$")


def parse_tp_runs():
    """{(boards, run): dict} from rank 0's summary line in each '===== tpN-i' section."""
    runs, cur = {}, None
    for line in read_lines("0013-tp-runs.txt"):
        m = re.match(r"^===== tp(\d)-(\d)$", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)))
            continue
        m = RUN_RE.match(line)
        if m and cur and cur not in runs:
            runs[cur] = dict(prompt=int(m.group(1)), generated=int(m.group(2)), gen_s=float(m.group(3)),
                             tps=m.group(4), engine=m.group(5), exchange=m.group(6), cpu=m.group(7))
    assert sorted(runs) == [(b, i) for b in (1, 2, 4) for i in (1, 2, 3)], sorted(runs)
    for r in runs.values():   # the captions say so; the logs must agree
        assert r["prompt"] == 24 and r["generated"] == 400, r
    return runs


def parse_phases():
    out = {}
    for line in read_lines("0013-phases.txt"):
        if line.startswith("#") or not line.strip():
            continue
        name, a, b = line.split("\t")
        out[name] = (float(a), float(b))
    return out


def parse_power():
    out = {}
    for i, host in enumerate(HOSTS, 1):
        rows = [l.split() for l in read_lines(f"0013-power-kv260-0{i}.txt") if l.strip() and not l.startswith("#")]
        out[host] = [(float(t), float(w)) for t, w in rows]
    return out


def mean_power(power, host, a, b):
    v = [w for t, w in power[host] if a <= t <= b]
    assert len(v) > 20, (host, a, b, len(v))
    return st.mean(v)


def parse_report():
    """The session report's own figures, for the cross-check."""
    tp, cpu_w = {}, None
    for line in read_lines("0013-session-report.txt"):
        m = re.match(r"^tp(\d)-(\d): .*decode power of the \d running boards ([\d.]+) W, ([\d.]+) J/token$", line)
        if m:
            tp[(int(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
        m = re.match(r"^cpu-llama-bench: \d+ s; power per board: ([\d.]+) ", line)
        if m:
            cpu_w = float(m.group(1))
    return tp, cpu_w


def parse_llama_bench():
    """The successful llama-bench attempt: mean and s.d. of tg400 on kv260-01."""
    lines = read_lines("0013-cpu-llama-bench.txt")
    cmd = None
    for line in lines:
        if line.startswith("### ssh"):
            cmd = line
        m = re.match(r"^\|.*\|\s*tg(\d+)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|$", line)
        if m:
            for flag in (f"-n {m.group(1)}", "-r 3", "-t 4", "-ngl 0", "-dev none", "-p 0"):   # the caption's conditions
                assert flag in cmd, (flag, cmd)
            return dict(n=int(m.group(1)), tps=m.group(2), sd=m.group(3), cmd=cmd)
    raise SystemExit("no tg400 row in 0013-cpu-llama-bench.txt")


def load():
    runs = parse_tp_runs()
    phases, power = parse_phases(), parse_power()
    report, report_cpu_w = parse_report()
    for (boards, i), r in runs.items():
        a, b = phases[f"tp{boards}-{i}"]
        w = sum(mean_power(power, h, b - r["gen_s"] + 1.5, b - 2.5) for h in HOSTS[:boards])
        r["watts"], r["jpt"] = w, w / float(r["tps"])
        r["date"] = datetime.datetime.fromtimestamp(b, datetime.timezone.utc).date().isoformat()
        rw, rj = report[(boards, i)]
        assert abs(w - rw) < 0.0006 and abs(r["jpt"] - rj) < 0.00006, ((boards, i), w, rw, r["jpt"], rj)
    cpu = parse_llama_bench()
    a, b = phases["cpu-llama-bench"]
    cpu["watts"] = mean_power(power, HOSTS[0], a + 30, b)
    assert abs(cpu["watts"] - report_cpu_w) < 0.0006, (cpu["watts"], report_cpu_w)
    cpu["jpt"] = cpu["watts"] / float(cpu["tps"])
    return runs, cpu


def cond(runs):
    """The run conditions the captions state, from the logs (one value across all runs)."""
    (p,), (n,), (date,) = ({r[k] for r in runs.values()} for k in ("prompt", "generated", "date"))
    return dict(p=p, n=n, pos=f"{p}-{p + n - 1}", date=date)


def rng(vals, fmt):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}-{hi}"


# ---------------------------------------------------------------- figure: token
def token_figure(runs):
    W = 800
    x0, x1 = 160, 680
    kinds = [("engine", ENGINE, "Engine"), ("exchange", EXCHANGE, "Exchange"), ("cpu", CPU, "CPU")]
    totals = [sum(float(r[k]) for k, _, _ in kinds) for r in runs.values()]
    xmax = math.ceil(max(totals) / 10) * 10
    sx = lambda v: x0 + (x1 - x0) * v / xmax
    bh, gap, ggap, top = 18, 5, 26, 104
    body = [text(20, 30, "Where one token's time goes", 18, INK, bold=True),
            text(20, 51, "Time per generated token on board 1 (rank 0), three runs for each board count", 13, INK2)]
    body += legend([("box", ENGINE, "Engine (matrix-vector calls on the FPGA)"),
                    ("box", EXCHANGE, "Exchange (network)"),
                    ("box", CPU, "CPU (one A53 core)")], 20, 80)
    ybot = top + 3 * (3 * bh + 2 * gap) + 2 * ggap
    for t in range(0, xmax + 1, 10):   # gridlines first, behind the bars
        body.append(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{ybot + 4}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(sx(t), ybot + 22, f"{t}", 12, MUTED, "middle"))
    body.append(f'<line x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{ybot + 4}" stroke="{AXIS}" stroke-width="1"/>')
    body.append(text((x0 + x1) / 2, ybot + 42, "milliseconds per generated token", 12.5, INK2, "middle"))
    desc = []
    y = top
    for boards in (1, 2, 4):
        gy = y
        label = "1 board" if boards == 1 else f"{boards} boards"
        body.append(text(20, gy + (3 * bh + 2 * gap) / 2 + 5, label, 13.5, INK, bold=True))
        # a term too thin to see in every run of the group is named under the group label
        thin = [(k, name) for k, _, name in kinds if all(sx(float(runs[(boards, i)][k])) - x0 < 6 for i in (1, 2, 3))]
        for j, (k, name) in enumerate(thin):
            vals = sorted({runs[(boards, i)][k] for i in (1, 2, 3)}, key=float)
            body.append(text(20, gy + (3 * bh + 2 * gap) / 2 + 22 + 14 * j, f"{name.lower()} {'-'.join(vals)} ms", 11.5, INK2))
        for i in (1, 2, 3):
            r = runs[(boards, i)]
            body.append(text(x0 - 10, y + bh / 2 + 4.5, f"run {i}", 11.5, INK2, "end"))
            xs = 0.0
            segs = [(k, colour, name, float(r[k]), r[k]) for k, colour, name in kinds]
            for n, (k, colour, name, v, vs) in enumerate(segs):
                xa, xb = sx(xs), sx(xs + v)
                left = xa + (1 if n else 0)
                right = xb - (1 if n < len(segs) - 1 else 0)
                tip = f"{label}, run {i}: {name.lower()} {vs} ms per token"
                if right - left > 0.5:
                    if n == len(segs) - 1:
                        body.append(bar_end(left, y, right - left, bh, 4, colour, tip))
                    else:
                        body.append(f'<rect x="{left:.2f}" y="{y}" width="{right - left:.2f}" height="{bh}" fill="{colour}"><title>{esc(tip)}</title></rect>')
                if right - left >= tw(vs, 11.5) + 10:
                    body.append(text((left + right) / 2, y + bh / 2 + 4.2, vs, 11.5, ON_FILL[colour], "middle"))
                xs += v
            body.append(text(sx(xs) + 8, y + bh / 2 + 4.5, f"{r['tps']} tokens/s", 12, INK2, extra=HALO))
            desc.append(f"{label}, run {i}: {r['tps']} tokens/s; engine {r['engine']} ms, exchange {r['exchange']} ms, CPU {r['cpu']} ms per token")
            y += bh + gap
        y += ggap - gap
    cy = ybot + 72
    c = cond(runs)
    cap = [f"Llama-3.2-1B-Instruct, every weight matrix ggml Q4_0. run-tp.sh, {c['n']} tokens generated after a {c['p']}-token prompt",
           f"(positions {c['pos']}), greedy; gemv2 bitstream at 250 MHz; rank 0's buckets: all {c['n'] + c['p'] - 1} forwards,",
           f"prompt included, over the {c['n']} tokens (about 6% high). Engine time includes preparing each call's input on the CPU.",
           f"{c['date']}, buildlog 0013; source: evidence/0013-tp-runs.txt."]
    for n, c in enumerate(cap):
        body.append(text(20, cy + 16 * n, c, 12, INK2))
    H = int(cy + 16 * (len(cap) - 1) + 14)
    return svg_doc(W, H, "Where one token's time goes on 1, 2 and 4 boards", "; ".join(desc) + ". " + " ".join(cap), body)


# ---------------------------------------------------------------- figure: scaling
def scaling_figure(runs, cpu):
    W = 800
    med = {b: st.median(float(runs[(b, i)]["tps"]) for i in (1, 2, 3)) for b in (1, 2, 4)}
    body = [text(20, 30, "Rate and energy per token on 1, 2 and 4 boards", 18, INK, bold=True),
            text(20, 51, "Each dot is one 400-token run; the open diamond is the same board without its engine (different software)", 13, INK2)]
    body += legend([("dot", ENGINE, "this project: runtime + FPGA engine, each run"),
                    ("diamond", BASELINE, "same board, CPU only: llama.cpp, mean of 3 runs")], 20, 80)
    top, ph = 132, 196
    bot = top + ph
    panels = [
        dict(x0=78, x1=372, key="tps", step=10, fmt="{:.0f}", title="Decode rate", unit="tokens per second"),
        dict(x0=488, x1=782, key="jpt", step=0.5, fmt="{:.1f}", title="Energy per generated token", unit="joules per token (module power)"),
    ]
    vals = {"tps": [float(r["tps"]) for r in runs.values()] + [float(cpu["tps"])],
            "jpt": [r["jpt"] for r in runs.values()] + [cpu["jpt"]]}
    desc = []
    for p in panels:
        key = p["key"]
        ymax = math.ceil(max(vals[key]) * 1.08 / p["step"]) * p["step"]
        sy = lambda v: bot - ph * v / ymax
        x0, x1 = p["x0"], p["x1"]
        body.append(text(x0 - 58, top - 22, p["title"], 14, INK, bold=True))
        t = 0.0
        while t <= ymax + 1e-9:
            body.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x1}" y2="{sy(t):.1f}" stroke="{GRID if t else AXIS}" stroke-width="1"/>')
            body.append(text(x0 - 8, sy(t) + 4, p["fmt"].format(t), 12, MUTED, "end"))
            t += p["step"]
        body.append(text(x0 - 40, (top + bot) / 2, p["unit"], 12.5, INK2, "middle",
                         extra=f' transform="rotate(-90 {x0 - 40} {(top + bot) / 2:.1f})"'))
        cx = {b: x0 + (x1 - x0) * f for b, f in ((1, 1 / 6), (2, 3 / 6), (4, 5 / 6))}
        for b in (1, 2, 4):
            label = "1 board" if b == 1 else f"{b} boards"
            body.append(text(cx[b], bot + 20, label, 12.5, INK, "middle"))
            rs = [runs[(b, i)] for i in (1, 2, 3)]
            if key == "jpt":
                body.append(text(cx[b], bot + 36, rng([r["watts"] for r in rs], "{:.2f}") + " W", 11.5, INK2, "middle"))
            ys = []
            for i, r in zip((1, 2, 3), rs):
                v = float(r["tps"]) if key == "tps" else r["jpt"]
                shown = r["tps"] if key == "tps" else f"{r['jpt']:.3f} J at {r['watts']:.2f} W"
                ys.append(sy(v))
                body.append(f'<circle cx="{cx[b] + (i - 2) * 10:.1f}" cy="{sy(v):.1f}" r="4.5" fill="{ENGINE}" stroke="{SURFACE}" stroke-width="2">'
                            f'<title>{esc(f"{label}, run {i}: {shown}")}</title></circle>')
            lab = rng([float(r["tps"]) for r in rs], "{:.2f}") if key == "tps" else rng([r["jpt"] for r in rs], "{:.3f}") + " J"
            body.append(text(cx[b], min(ys) - 11, lab, 12, INK2, "middle", extra=HALO))
        # the CPU-only baseline, same board count as one board, different software
        v = float(cpu["tps"]) if key == "tps" else cpu["jpt"]
        body.append(diamond(cx[1], sy(v), 6, BASELINE, f"CPU only: {cpu['tps']} ± {cpu['sd']} tokens/s, {cpu['watts']:.2f} W, {cpu['jpt']:.2f} J per token"))
        lab = f"CPU only: {cpu['tps']} ± {cpu['sd']}" if key == "tps" else f"CPU only: {cpu['jpt']:.2f} J at {cpu['watts']:.2f} W"
        body.append(text(cx[1] + 12, sy(v) + 4, lab, 12, INK2, extra=HALO))
    for b in (1, 2, 4):
        rs = [runs[(b, i)] for i in (1, 2, 3)]
        desc.append(f"{b} board(s): " + ", ".join(r["tps"] for r in rs) + " tokens/s; " +
                    ", ".join(f"{r['jpt']:.3f}" for r in rs) + " J per token at " + ", ".join(f"{r['watts']:.2f}" for r in rs) + " W")
    desc.append(f"CPU only: {cpu['tps']} ± {cpu['sd']} tokens/s, {cpu['watts']:.2f} W, {cpu['jpt']:.2f} J per token")
    s2, s4 = med[2] / med[1], med[4] / med[1]
    c = cond(runs)
    cy = bot + 66
    cap = [f"Llama-3.2-1B-Instruct, every weight matrix ggml Q4_0, {c['n']} generated tokens. Boards: run-tp.sh, {c['p']}-token prompt (positions",
           f"{c['pos']}), greedy, 3 runs each; medians {med[1]:.2f}, {med[2]:.2f}, {med[4]:.2f} tokens/s, {s2:.2f}x on two boards and {s4:.2f}x on four.",
           f"CPU only: llama-bench on board 1's four A53 cores, tg{cpu['n']} from an empty context (positions 0-{cpu['n'] - 1}), {cpu['tps']} ± {cpu['sd']} (mean ± s.d.).",
           "Power: the INA260 on each running K26 module, about 10 samples/s; carrier cards, fans, power supplies and the",
           f"switch are not measured. {c['date']}, buildlog 0013; source: evidence/0013-tp-runs.txt, -power-*, -cpu-llama-bench.txt."]
    for n, c in enumerate(cap):
        body.append(text(20, cy + 16 * n, c, 12, INK2))
    H = int(cy + 16 * (len(cap) - 1) + 14)
    return svg_doc(W, H, "Tokens per second and energy per token on 1, 2 and 4 boards", "; ".join(desc) + ". " + " ".join(cap), body)


# ---------------------------------------------------------------- main
def convert(name):
    svg = HERE / f"{name}.svg"
    env = dict(os.environ, SOURCE_DATE_EPOCH=os.environ.get("SOURCE_DATE_EPOCH", "1793750400"))   # stable PDF dates
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(HERE / f"{name}.pdf"), str(svg)], check=True, env=env)
    subprocess.run(["rsvg-convert", "-f", "png", "-z", "2", "-o", str(HERE / f"{name}.png"), str(svg)], check=True, env=env)


def main():
    runs, cpu = load()
    (HERE / "token.svg").write_text(token_figure(runs))
    (HERE / "scaling.svg").write_text(scaling_figure(runs, cpu))
    for name in ("system", "engine", "token", "scaling"):
        if (HERE / f"{name}.svg").exists():
            convert(name)
        else:
            print(f"make_figs: {name}.svg missing, skipped", file=sys.stderr)
    for (b, i), r in sorted(runs.items()):
        print(f"{b} board(s) run {i}: {r['tps']} tokens/s, engine {r['engine']} / exchange {r['exchange']} / cpu {r['cpu']} ms, "
              f"{r['watts']:.3f} W, {r['jpt']:.4f} J/token")
    print(f"CPU only: {cpu['tps']} ± {cpu['sd']} tokens/s, {cpu['watts']:.3f} W, {cpu['jpt']:.4f} J/token")


if __name__ == "__main__":
    main()
