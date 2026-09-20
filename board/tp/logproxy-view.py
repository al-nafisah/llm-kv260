#!/usr/bin/env python3
"""
logproxy-view — a local dashboard for the logproxy JSONL.

    python3 board/tp/logproxy-view.py          # then open http://127.0.0.1:8090

Binds 127.0.0.1 only and holds the data in this process: the log contains
visitors' IP addresses and prompt text, so none of it leaves the machine.
Reads the live log plus any rotated .1 ... .5 files.

Environment:
  LISTEN    127.0.0.1:8090
  LOG_FILE  ~/.local/share/kv260-logproxy/requests.jsonl
"""

import http.server
import json
import os
import socketserver
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

LISTEN = os.environ.get("LISTEN", "127.0.0.1:8090")
LISTEN_HOST, _, _p = LISTEN.partition(":")
LISTEN_PORT = int(_p or 8090)
LOG_FILE = os.path.expanduser(
    os.environ.get("LOG_FILE", "~/.local/share/kv260-logproxy/requests.jsonl"))

RECENT_MAX = 300


def read_records():
    """Oldest file first, so the timeline runs forward."""
    paths = [f"{LOG_FILE}.{i}" for i in range(5, 0, -1)] + [LOG_FILE]
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def parse_ts(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def place(r):
    city, country = r.get("city"), r.get("country")
    if city and country:
        return f"{city}, {country}"
    return country or "unknown"


def summarize(recs):
    reqs = [r for r in recs if r.get("kind") == "request"]
    polls = [r for r in recs if r.get("kind") == "poll_rollup"]
    rejected = [r for r in recs if r.get("kind") in ("rejected", "error")]
    prompts = [r for r in reqs if r.get("prompt")]

    ips = {r.get("ip") for r in recs if r.get("ip") and r["ip"] != "?"}
    tps = [r["tps"] for r in reqs if isinstance(r.get("tps"), (int, float))]
    toks = sum(r.get("gen_tokens") or 0 for r in reqs)

    # --- timeline: hourly if the span is short, else daily ----------------
    stamps = [t for t in (parse_ts(r.get("ts")) for r in reqs) if t]
    timeline, bucket = [], "hour"
    if stamps:
        span = max(stamps) - min(stamps)
        bucket = "hour" if span <= timedelta(days=3) else "day"
        counts = Counter()
        for t in stamps:
            key = (t.replace(minute=0, second=0, microsecond=0) if bucket == "hour"
                   else t.replace(hour=0, minute=0, second=0, microsecond=0))
            counts[key] += 1
        step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
        cur, end = min(counts), max(counts)
        while cur <= end:                      # fill gaps so time reads evenly
            timeline.append({
                "t": cur.strftime("%Y-%m-%dT%H:%MZ"),
                "label": cur.strftime("%H:%M" if bucket == "hour" else "%d %b"),
                "full": cur.strftime("%d %b %H:%M UTC" if bucket == "hour"
                                     else "%d %b %Y"),
                "n": counts.get(cur, 0)})
            cur += step
        timeline = timeline[-96:]

    # --- places: prompts and pollers, kept apart --------------------------
    by_place = defaultdict(lambda: {"prompts": 0, "visits": 0, "ips": set()})
    for r in reqs:
        e = by_place[place(r)]
        e["prompts"] += 1
        if r.get("ip") and r["ip"] != "?":
            e["ips"].add(r["ip"])
    for r in polls:
        e = by_place[place(r)]
        e["visits"] += r.get("n", 0)
        if r.get("ip") and r["ip"] != "?":
            e["ips"].add(r["ip"])
    places = sorted(
        ({"place": k, "prompts": v["prompts"], "visits": v["visits"],
          "ips": len(v["ips"])} for k, v in by_place.items()),
        key=lambda d: (-d["prompts"], -d["visits"], d["place"]))

    recent = []
    for r in reversed(prompts[-RECENT_MAX:]):
        recent.append({
            "ts": r.get("ts"), "place": place(r), "ip": r.get("ip", "?"),
            "prompt": r.get("prompt", ""), "reply": r.get("reply", ""),
            "tokens": r.get("gen_tokens"), "tps": r.get("tps"),
            "status": r.get("status"), "streamed": r.get("streamed", False),
            "ua": r.get("ua", ""), "dropped": r.get("client_disconnected", False)})

    return {
        "log_file": LOG_FILE,
        "records": len(recs),
        "generated": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        "kpi": {
            "prompts": len(prompts),
            "visitors": len(ips),
            "countries": len({r.get("country") for r in recs if r.get("country")}),
            "tokens": toks,
            "median_tps": round(statistics.median(tps), 1) if tps else None,
            "rejected": len(rejected),
            "dropped": sum(1 for r in reqs if r.get("client_disconnected")),
        },
        "bucket": bucket,
        "timeline": timeline,
        "places": places[:12],
        "recent": recent,
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV260 API — request log</title>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f4f3f0;  --surface-1: #fcfcfb;  --border: #e2e1dc;
    --text-primary: #0b0b0b;  --text-secondary: #52514e;  --text-muted: #85837c;
    --series-1: #2a78d6;  --series-1-soft: #9ec5f4;
    --grid: #ebeae6;  --good: #1baf7a;  --warn: #eda100;  --bad: #e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #121211;  --surface-1: #1a1a19;  --border: #333330;
      --text-primary: #ffffff;  --text-secondary: #c3c2b7;  --text-muted: #8f8e85;
      --series-1: #3987e5;  --series-1-soft: #1c5cab;
      --grid: #2a2a28;  --good: #199e70;  --warn: #c98500;  --bad: #e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #121211;  --surface-1: #1a1a19;  --border: #333330;
    --text-primary: #ffffff;  --text-secondary: #c3c2b7;  --text-muted: #8f8e85;
    --series-1: #3987e5;  --series-1-soft: #1c5cab;
    --grid: #2a2a28;  --good: #199e70;  --warn: #c98500;  --bad: #e66767;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    padding-block: 24px; padding-left: 16px; padding-right: 16px;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  header { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
           justify-content: space-between; margin-bottom: 20px; }
  h1 { font-size: 19px; margin: 0; font-weight: 620; letter-spacing: -0.01em; }
  .sub { color: var(--text-muted); font-size: 12.5px; }
  .controls { display: flex; gap: 14px; align-items: center; font-size: 12.5px;
              color: var(--text-secondary); }
  .controls label { display: flex; gap: 6px; align-items: center; cursor: pointer; }
  .card { background: var(--surface-1); border: 1px solid var(--border);
          border-radius: 10px; padding: 16px 18px; }
  .kpis { display: grid; gap: 10px; margin-bottom: 14px;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
  .kpi .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em;
                color: var(--text-muted); margin-bottom: 6px; }
  .kpi .value { font-size: 27px; font-weight: 600; letter-spacing: -0.02em;
                font-variant-numeric: tabular-nums; }
  .kpi .unit { font-size: 13px; color: var(--text-muted); font-weight: 400;
               margin-left: 3px; }
  .grid2 { display: grid; gap: 14px; grid-template-columns: 1fr 1fr;
           margin-bottom: 14px; }
  @media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
  h2 { font-size: 13px; margin: 0 0 2px; font-weight: 600; }
  .hint { font-size: 12px; color: var(--text-muted); margin: 0 0 12px; }
  svg { display: block; width: 100%; overflow: visible; }
  .tip { position: fixed; pointer-events: none; z-index: 9; opacity: 0;
         transition: opacity .08s; background: var(--surface-1);
         border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px;
         font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.14);
         color: var(--text-primary); max-width: 300px; }
  .tip b { font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-weight: 600; font-size: 11.5px; letter-spacing: .05em;
       text-transform: uppercase; color: var(--text-muted);
       border-bottom: 1px solid var(--border); padding: 0 10px 8px 0; }
  td { padding: 9px 10px 9px 0; border-bottom: 1px solid var(--grid);
       vertical-align: top; }
  tbody tr:last-child td { border-bottom: none; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
          color: var(--text-secondary); white-space: nowrap; }
  .num { font-variant-numeric: tabular-nums; text-align: right;
         white-space: nowrap; color: var(--text-secondary); }
  .prompt { color: var(--text-primary); }
  .reply { color: var(--text-muted); font-size: 12px; margin-top: 3px;
           display: none; }
  body.replies .reply { display: block; }
  .tablewrap { overflow-x: auto; }
  .empty { color: var(--text-muted); text-align: center; padding: 34px 10px;
           font-size: 13px; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
         margin-right: 5px; vertical-align: 1px; }
  footer { color: var(--text-muted); font-size: 12px; margin-top: 18px; }
  code { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>KV260 API — request log</h1>
      <div class="sub" id="sub">loading…</div>
    </div>
    <div class="controls">
      <label><input type="checkbox" id="maskIPs" checked> mask IPs</label>
      <label><input type="checkbox" id="showReplies"> show replies</label>
    </div>
  </header>

  <div class="kpis" id="kpis"></div>

  <div class="grid2">
    <div class="card">
      <h2>Prompts over time</h2>
      <p class="hint" id="timeHint">—</p>
      <div id="timeChart"></div>
    </div>
    <div class="card">
      <h2>Where they came from</h2>
      <p class="hint">Prompts per location. Country is reliable; city is a guess from the IP.</p>
      <div id="placeChart"></div>
    </div>
  </div>

  <div class="card">
    <h2>Recent prompts</h2>
    <p class="hint">Newest first. Hover a row for the user agent.</p>
    <div class="tablewrap"><table>
      <thead><tr>
        <th>When (UTC)</th><th>Where</th><th>Prompt</th>
        <th class="num">Tokens</th><th class="num">t/s</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
    <div class="empty" id="emptyRows" hidden>No prompts logged yet.</div>
  </div>

  <footer id="foot"></footer>
</div>
<div class="tip" id="tip"></div>

<script>
const tip = document.getElementById('tip');
let DATA = null;

function showTip(html, ev) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY - r.height - 10;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y < 8) y = ev.clientY + 16;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => tip.style.opacity = 0;

function maskIP(ip) {
  if (!document.getElementById('maskIPs').checked) return ip;
  if (!ip || ip === '?') return '?';
  if (ip.includes(':')) return ip.split(':').slice(0, 2).join(':') + ':…';
  const p = ip.split('.');
  return p.length === 4 ? p[0] + '.' + p[1] + '.x.x' : ip;
}
const esc = s => (s ?? '').toString().replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* a rect with only its data-end rounded, anchored to the baseline */
function colPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y}
          H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}
function barPath(x, y, w, h, r) {
  r = Math.min(r, h / 2, w);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r}
          V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}

function drawTimeline() {
  const host = document.getElementById('timeChart');
  const d = DATA.timeline;
  if (!d.length) { host.innerHTML = '<div class="empty">Nothing logged yet.</div>'; return; }
  const W = host.clientWidth || 480, H = 190;
  const m = {t: 8, r: 6, b: 24, l: 30};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const max = Math.max(1, ...d.map(o => o.n));
  const gap = 2;
  /* cap the width so a handful of points don't render as slabs, then centre */
  const bw = Math.min(40, Math.max(1, iw / d.length - gap));
  const x0 = m.l + Math.max(0, (iw - (d.length * (bw + gap) - gap)) / 2);
  const ticks = [0, Math.ceil(max / 2), max].filter((v, i, a) => a.indexOf(v) === i);

  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Prompts over time">`;
  for (const t of ticks) {
    const y = m.t + ih - (t / max) * ih;
    s += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}"
           stroke="var(--grid)" stroke-width="1"/>
          <text x="${m.l - 7}" y="${y + 4}" text-anchor="end" font-size="10.5"
           fill="var(--text-muted)">${t}</text>`;
  }
  d.forEach((o, i) => {
    const h = o.n === 0 ? 0 : Math.max(2, (o.n / max) * ih);
    const x = x0 + i * (bw + gap), y = m.t + ih - h;
    if (h) s += `<path d="${colPath(x, y, bw, h, 4)}" fill="var(--series-1)"
                  data-i="${i}"/>`;
  });
  const step = Math.ceil(d.length / 6);
  d.forEach((o, i) => {
    if (i % step) return;
    s += `<text x="${x0 + i * (bw + gap) + bw / 2}" y="${H - 7}"
           text-anchor="middle" font-size="10.5" fill="var(--text-muted)">${o.label}</text>`;
  });
  /* transparent hit strips, wider than the marks */
  d.forEach((o, i) => {
    s += `<rect x="${x0 + i * (bw + gap) - gap / 2}" y="${m.t}"
           width="${bw + gap}" height="${ih}" fill="transparent" data-hit="${i}"/>`;
  });
  s += '</svg>';
  host.innerHTML = s;
  host.querySelectorAll('[data-hit]').forEach(el => {
    const o = d[+el.dataset.hit];
    el.addEventListener('mousemove', e => showTip(
      `${o.full}<br><b>${o.n}</b> prompt${o.n === 1 ? '' : 's'}`, e));
    el.addEventListener('mouseleave', hideTip);
  });
}

function drawPlaces() {
  const host = document.getElementById('placeChart');
  const d = DATA.places.filter(p => p.prompts > 0);
  if (!d.length) { host.innerHTML = '<div class="empty">No prompts yet.</div>'; return; }
  const W = host.clientWidth || 480;
  const rowH = 26, H = d.length * rowH + 6;
  const labelW = Math.min(150, Math.max(70, W * 0.32));
  const valW = 42, iw = W - labelW - valW;
  const max = Math.max(1, ...d.map(o => o.prompts));

  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Prompts by location">`;
  d.forEach((o, i) => {
    const y = i * rowH + 3, bh = 15;
    const w = Math.max(2, (o.prompts / max) * iw);
    s += `<text x="0" y="${y + bh - 3}" font-size="12.5" fill="var(--text-secondary)">
            ${esc(o.place).slice(0, 22)}</text>
          <path d="${barPath(labelW, y, w, bh, 4)}" fill="var(--series-1)"/>
          <text x="${labelW + w + 8}" y="${y + bh - 3}" font-size="12"
            fill="var(--text-secondary)" font-variant-numeric="tabular-nums">${o.prompts}</text>
          <rect x="0" y="${y - 3}" width="${W}" height="${rowH}" fill="transparent"
            data-p="${i}"/>`;
  });
  s += '</svg>';
  host.innerHTML = s;
  host.querySelectorAll('[data-p]').forEach(el => {
    const o = d[+el.dataset.p];
    el.addEventListener('mousemove', e => showTip(
      `<b>${esc(o.place)}</b><br><b>${o.prompts}</b> prompt${o.prompts === 1 ? '' : 's'}
       · <b>${o.ips}</b> address${o.ips === 1 ? '' : 'es'}
       ${o.visits ? `<br><b>${o.visits}</b> page polls` : ''}`, e));
    el.addEventListener('mouseleave', hideTip);
  });
}

function drawKPIs() {
  const k = DATA.kpi;
  const tile = (label, value, unit = '') => `<div class="card kpi">
      <div class="label">${label}</div>
      <div class="value">${value}${unit ? `<span class="unit">${unit}</span>` : ''}</div>
    </div>`;
  let h = tile('Prompts', k.prompts)
        + tile('Unique addresses', k.visitors)
        + tile('Countries', k.countries)
        + tile('Tokens generated', k.tokens.toLocaleString())
        + tile('Median rate', k.median_tps ?? '—', k.median_tps ? ' t/s' : '');
  if (k.rejected) h += tile('Rejected', k.rejected);
  if (k.dropped) h += tile('Client hung up', k.dropped);
  document.getElementById('kpis').innerHTML = h;
}

function drawRows() {
  const tb = document.getElementById('rows');
  const empty = document.getElementById('emptyRows');
  empty.hidden = DATA.recent.length > 0;
  tb.innerHTML = DATA.recent.map(r => {
    const bad = r.status && r.status >= 400;
    const when = (r.ts || '').replace('T', ' ').replace('Z', '');
    return `<tr title="${esc(r.ua)}">
      <td class="mono">${esc(when)}</td>
      <td class="mono">${bad ? `<span class="dot" style="background:var(--bad)"></span>` : ''}
          ${esc(r.place)}<br><span style="color:var(--text-muted)">${esc(maskIP(r.ip))}</span></td>
      <td><div class="prompt">${esc(r.prompt)}</div>
          ${r.reply ? `<div class="reply">→ ${esc(r.reply)}</div>` : ''}
          ${r.dropped ? `<div class="reply" style="display:block;color:var(--warn)">
             client disconnected mid-stream</div>` : ''}</td>
      <td class="num">${r.tokens ?? '—'}</td>
      <td class="num">${r.tps ?? (r.streamed ? 'stream' : '—')}</td>
    </tr>`;
  }).join('');
}

function render() {
  document.getElementById('sub').textContent =
    `${DATA.records} records · ${DATA.log_file} · updated ${DATA.generated}`;
  document.getElementById('timeHint').textContent =
    DATA.bucket === 'hour' ? 'One column per hour.' : 'One column per day.';
  document.getElementById('foot').innerHTML =
    `Local only — served on 127.0.0.1, nothing leaves this machine.
     Refreshes every 30 s. Stop with <code>Ctrl-C</code> in the terminal.`;
  drawKPIs(); drawTimeline(); drawPlaces(); drawRows();
}

async function load() {
  try {
    DATA = await (await fetch('data.json', {cache: 'no-store'})).json();
    render();
  } catch (e) {
    document.getElementById('sub').textContent = 'could not read the log: ' + e;
  }
}
document.getElementById('maskIPs').addEventListener('change', drawRows);
document.getElementById('showReplies').addEventListener('change', e =>
  document.body.classList.toggle('replies', e.target.checked));
addEventListener('resize', () => { if (DATA) { drawTimeline(); drawPlaces(); } });
load();
setInterval(load, 30000);
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/data.json":
            try:
                data = summarize(read_records())
            except Exception as exc:                      # never 500 the page
                data = {"records": 0, "log_file": LOG_FILE, "generated": "",
                        "kpi": {"prompts": 0, "visitors": 0, "countries": 0,
                                "tokens": 0, "median_tps": None, "rejected": 0,
                                "dropped": 0},
                        "bucket": "hour", "timeline": [], "places": [],
                        "recent": [], "error": str(exc)}
            self._send(json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    if not os.path.exists(LOG_FILE):
        print(f"note: {LOG_FILE} does not exist yet — the page will be empty "
              f"until the proxy logs something.", file=sys.stderr)
    srv = Server((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"logproxy-view: http://{LISTEN_HOST}:{LISTEN_PORT}  "
          f"(reading {LOG_FILE})\nCtrl-C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
