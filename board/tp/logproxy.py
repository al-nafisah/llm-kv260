#!/usr/bin/env python3
"""
logproxy — a logging reverse proxy that sits between cloudflared and board 1.

    browser -> Cloudflare -> cloudflared (this PC) -> logproxy :8081 -> 10.42.0.137:8080

It records who asked what: the Cloudflare-injected client IP and geo headers,
the user agent, the prompt, and how many tokens came back.  Nothing about the
response changes, so the requester sees no difference.

Responses stream through chunk by chunk, so SSE token streaming still arrives
token by token.  The polling endpoints (/health, /stats, /v1/models) are not
written per request -- the page polls them from every visitor's browser and
would bury the prompts -- they are counted and flushed as one rollup record
every ROLLUP_MIN minutes, which is the better signal for visitors anyway.

The log is JSONL at LOG_FILE, mode 0600, rotated at ROTATE_MB.

Environment:
  LISTEN            127.0.0.1:8081     where cloudflared points
  UPSTREAM          10.42.0.137:8080   board 1
  LOG_FILE          ~/.local/share/kv260-logproxy/requests.jsonl
  ROTATE_MB         64                 rotate at this size, keeping ROTATE_KEEP
  ROTATE_KEEP       5
  MAX_BODY_KB       1024               refuse a larger body with 413; 0 disables
  PROMPT_MAX        4096               characters kept per logged text field
  HASH_IPS          0                  1 = log a salted hash instead of the IP
  LOG_POLLS         0                  1 = log every /health, /stats, /v1/models
  ROLLUP_MIN        60                 minutes between poll rollup records
  UPSTREAM_TIMEOUT  300                seconds
"""

import hashlib
import http.client
import http.server
import json
import os
import secrets
import signal
import socket
import socketserver
import sys
import threading
import time
from datetime import datetime, timezone


def _env(name, default):
    return os.environ.get(name, default)


def _split_hostport(s, default_port):
    host, _, port = s.partition(":")
    return host, int(port or default_port)


LISTEN_HOST, LISTEN_PORT = _split_hostport(_env("LISTEN", "127.0.0.1:8081"), 8081)
UP_HOST, UP_PORT = _split_hostport(_env("UPSTREAM", "10.42.0.137:8080"), 8080)
LOG_FILE = os.path.expanduser(
    _env("LOG_FILE", "~/.local/share/kv260-logproxy/requests.jsonl"))
ROTATE_BYTES = int(_env("ROTATE_MB", "64")) * 1024 * 1024
ROTATE_KEEP = int(_env("ROTATE_KEEP", "5"))
MAX_BODY = int(_env("MAX_BODY_KB", "1024")) * 1024
PROMPT_MAX = int(_env("PROMPT_MAX", "4096"))
HASH_IPS = _env("HASH_IPS", "0") == "1"
LOG_POLLS = _env("LOG_POLLS", "0") == "1"
ROLLUP_SEC = int(_env("ROLLUP_MIN", "60")) * 60
UPSTREAM_TIMEOUT = int(_env("UPSTREAM_TIMEOUT", "300"))

POLL_PATHS = ("/health", "/stats", "/v1/models")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "te",
              "proxy-authorization", "trailers", "transfer-encoding", "upgrade"}

# Cloudflare adds the first two on every proxied request; the rest arrive once
# "Add visitor location headers" is on under Rules -> Managed Transforms.
GEO_HEADERS = {
    "country": "cf-ipcountry", "city": "cf-ipcity", "region": "cf-region",
    "tz": "cf-timezone", "lat": "cf-iplatitude", "lon": "cf-iplongitude",
    "continent": "cf-ipcontinent", "postal": "cf-postal-code",
}

SALT = ""
_write_lock = threading.Lock()
_poll_lock = threading.Lock()
_polls = {}


class TooBig(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _salt():
    """A per-install salt, so a hashed IP cannot be reversed with a rainbow table."""
    path = os.path.join(os.path.dirname(LOG_FILE), "ip-salt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    s = secrets.token_hex(16)
    with open(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600), "w") as f:
        f.write(s)
    return s


def _rotate():
    try:
        if os.path.getsize(LOG_FILE) < ROTATE_BYTES:
            return
    except OSError:
        return
    for i in range(ROTATE_KEEP - 1, 0, -1):
        older, newer = f"{LOG_FILE}.{i + 1}", f"{LOG_FILE}.{i}"
        if os.path.exists(newer):
            os.replace(newer, older)
    os.replace(LOG_FILE, f"{LOG_FILE}.1")


def emit(rec):
    line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _write_lock:
        _rotate()
        fd = os.open(LOG_FILE, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(line)


def clip(s, n=PROMPT_MAX):
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n} chars]"


def client_id(headers):
    ip = (headers.get("CF-Connecting-IP")
          or (headers.get("X-Forwarded-For") or "").split(",")[0].strip()
          or "?")
    if HASH_IPS and ip != "?":
        return hashlib.sha256((SALT + ip).encode()).hexdigest()[:16]
    return ip


def geo_of(headers):
    g = {}
    for key, header in GEO_HEADERS.items():
        v = headers.get(header)
        if v:
            g[key] = v
    return g


def describe_body(body):
    """Pull the interesting fields out of an OpenAI-shaped request body."""
    out = {}
    try:
        b = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        if body:
            out["body_unparsed"] = clip(body.decode("utf-8", "replace"), 512)
        return out
    if not isinstance(b, dict):
        return out
    for k in ("model", "stream", "max_tokens", "temperature", "top_p", "seed"):
        if k in b:
            out[k] = b[k]
    msgs = b.get("messages")
    if isinstance(msgs, list):
        out["n_messages"] = len(msgs)
        out["messages"] = [
            {"role": m.get("role"), "content": clip(m.get("content"))}
            for m in msgs if isinstance(m, dict)
        ]
        users = [m.get("content") for m in msgs
                 if isinstance(m, dict) and m.get("role") == "user"
                 and isinstance(m.get("content"), str)]
        if users:
            out["prompt"] = clip(users[-1])
    elif isinstance(b.get("prompt"), str):
        out["prompt"] = clip(b["prompt"])
    return out


def note_poll(ip, geo, path, ua):
    with _poll_lock:
        key = (ip, geo.get("country", ""), geo.get("city", ""))
        e = _polls.setdefault(key, {"n": 0, "paths": {}, "uas": set(),
                                    "first": _now(), "last": _now()})
        e["n"] += 1
        e["paths"][path] = e["paths"].get(path, 0) + 1
        e["uas"].add(clip(ua or "", 120))
        e["last"] = _now()


def flush_polls():
    with _poll_lock:
        items = list(_polls.items())
        _polls.clear()
    for (ip, country, city), e in items:
        emit({"ts": _now(), "kind": "poll_rollup", "ip": ip,
              "country": country, "city": city, "n": e["n"],
              "paths": e["paths"], "uas": sorted(e["uas"])[:5],
              "first": e["first"], "last": e["last"],
              "window_min": ROLLUP_SEC // 60})


def rollup_loop():
    while True:
        time.sleep(ROLLUP_SEC)
        try:
            flush_polls()
        except Exception as exc:
            print(f"rollup failed: {exc}", file=sys.stderr, flush=True)


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kv260-logproxy"

    def log_message(self, *a):
        pass  # the JSONL log is the record; keep the journal quiet

    # -- request body ----------------------------------------------------
    def _read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            out = bytearray()
            while True:
                line = self.rfile.readline(65536).split(b";")[0].strip()
                n = int(line or b"0", 16)
                if n == 0:
                    while True:
                        t = self.rfile.readline(65536)
                        if t in (b"\r\n", b"\n", b""):
                            break
                    break
                out += self.rfile.read(n)
                self.rfile.read(2)
                if MAX_BODY and len(out) > MAX_BODY:
                    raise TooBig()
            return bytes(out)
        n = int(self.headers.get("Content-Length") or 0)
        if MAX_BODY and n > MAX_BODY:
            raise TooBig()
        return self.rfile.read(n) if n else b""

    # -- the one handler for every method --------------------------------
    def _handle(self):
        t0 = time.time()
        ip = client_id(self.headers)
        geo = geo_of(self.headers)
        ua = self.headers.get("User-Agent", "")
        ray = self.headers.get("CF-Ray", "")
        path = self.path
        is_poll = self.command == "GET" and path.split("?")[0] in POLL_PATHS

        try:
            body = self._read_body()
        except TooBig:
            self.send_response_only(413)
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            emit({"ts": _now(), "kind": "rejected", "ip": ip, **geo,
                  "ua": clip(ua, 300), "ray": ray, "method": self.command,
                  "path": path, "status": 413, "reason": "body over MAX_BODY_KB",
                  "content_length": self.headers.get("Content-Length")})
            self.close_connection = True
            return

        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP}
        headers["Content-Length"] = str(len(body))

        try:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT,
                                              timeout=UPSTREAM_TIMEOUT)
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            payload = json.dumps({"error": "upstream unreachable"}).encode()
            self.send_response_only(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            emit({"ts": _now(), "kind": "error", "ip": ip, **geo,
                  "method": self.command, "path": path, "status": 502,
                  "error": str(exc)[:200]})
            return

        gen_tokens, extra, streamed = self._relay(resp)
        conn.close()

        if is_poll and not LOG_POLLS:
            note_poll(ip, geo, path.split("?")[0], ua)
            return

        rec = {"ts": _now(), "kind": "request", "ip": ip, **geo,
               "ua": clip(ua, 300), "ray": ray, "method": self.command,
               "path": path, "status": resp.status,
               "ms": round((time.time() - t0) * 1000, 1),
               "streamed": streamed}
        rec.update(describe_body(body))
        if gen_tokens is not None:
            rec["gen_tokens"] = gen_tokens
        if extra:
            rec.update(extra)
        emit(rec)

    def _relay(self, resp):
        """Forward status, headers and body. Returns (tokens, extra, streamed)."""
        out_headers = [(k, v) for k, v in resp.getheaders()
                       if k.lower() not in HOP_BY_HOP
                       and k.lower() != "content-length"]
        clen = resp.getheader("Content-Length")
        ctype = (resp.getheader("Content-Type") or "").lower()
        sse = "text/event-stream" in ctype

        self.send_response_only(resp.status)
        for k, v in out_headers:
            self.send_header(k, v)
        if clen is not None:
            self.send_header("Content-Length", clen)
            chunked = False
        else:
            self.send_header("Transfer-Encoding", "chunked")
            chunked = True
        self.end_headers()

        tokens = 0 if sse else None
        sse_buf = b""
        body_head = bytearray()
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                if chunked:
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()          # token by token, never buffered
                if sse:
                    sse_buf += chunk
                    parts = sse_buf.split(b"\n")
                    sse_buf = parts.pop()
                    for line in parts:
                        if line.startswith(b"data:"):
                            payload = line[5:].strip()
                            if payload and payload != b"[DONE]":
                                try:
                                    d = json.loads(payload)
                                    delta = d["choices"][0].get("delta", {})
                                    if delta.get("content"):
                                        tokens += 1
                                except Exception:
                                    pass
                elif len(body_head) < 262144:
                    body_head += chunk
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-stream. The board keeps generating --
            # that is the known unfixed hole, not something this proxy closes.
            self.close_connection = True
            return tokens, {"client_disconnected": True}, sse

        extra = {}
        if not sse and body_head:
            try:
                d = json.loads(bytes(body_head).decode("utf-8", "replace"))
                usage = d.get("usage") or {}
                if usage:
                    extra["gen_tokens"] = usage.get("completion_tokens")
                    extra["prompt_tokens"] = usage.get("prompt_tokens")
                tim = d.get("timings") or {}
                if tim.get("predicted_per_second"):
                    extra["tps"] = round(tim["predicted_per_second"], 2)
                ch = (d.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                if isinstance(msg.get("content"), str):
                    extra["reply"] = clip(msg["content"])
                if d.get("error"):
                    extra["upstream_error"] = clip(str(d["error"]), 300)
            except Exception:
                pass
        return tokens, extra, sse

    do_GET = do_POST = do_PUT = do_DELETE = _handle
    do_OPTIONS = do_HEAD = do_PATCH = _handle


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, socket.timeout)):
            return
        super().handle_error(request, client_address)


def main():
    os.makedirs(os.path.dirname(LOG_FILE), mode=0o700, exist_ok=True)
    try:
        os.chmod(os.path.dirname(LOG_FILE), 0o700)
    except OSError:
        pass
    global SALT
    SALT = _salt() if HASH_IPS else ""

    threading.Thread(target=rollup_loop, daemon=True).start()
    srv = Server((LISTEN_HOST, LISTEN_PORT), Proxy)

    # systemd stops us with SIGTERM; without this the counters since the last
    # rollup are lost on every restart.  serve_forever() runs off the main
    # thread so the handler can shut it down without deadlocking against it.
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    print(f"logproxy: {LISTEN_HOST}:{LISTEN_PORT} -> {UP_HOST}:{UP_PORT}, "
          f"log {LOG_FILE} (polls {'logged' if LOG_POLLS else 'rolled up'}, "
          f"ips {'hashed' if HASH_IPS else 'plain'})", flush=True)
    try:
        while not stopping.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        flush_polls()


if __name__ == "__main__":
    main()
