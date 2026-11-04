#!/usr/bin/env python3
"""Chat with the four boards from a terminal. Python 3 standard library only.

    python3 board/tp/chat.py                          # REPL, multi-turn
    python3 board/tp/chat.py "one prompt"             # one answer, then exit
    python3 board/tp/chat.py --url http://10.42.0.137:8080 --max-tokens 400 \\
        --system "Answer briefly." --temperature 0.7 --no-stream

REPL commands: /reset clears the conversation (the --system prompt stays),
/stats prints the boards, /quit exits (so does Ctrl-D). Ctrl-C during a
reply stops printing it; the server still runs that request to its end,
since nothing on the server side checks for a departed client, and the next
prompt waits behind it.

The API is tp_llama.cpp's (board/tp, section "server"): one request at a
time in arrival order, up to 8 waiting, HTTP 429 beyond that. A streamed
reply is SSE; its last chunk carries "timings" (prompt_n, prompt_ms,
predicted_n, predicted_ms, predicted_per_second), and the non-streaming
reply has the same object. After each reply this prints one status line:
tokens, rate and first-token time from those timings (client-side if the
server sent none), then the last chat's per-token split (engine + cpu +
exchange, the mean over the boards) and each board's power from /stats,
read right after the reply.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://10.42.0.137:8080"
CTX = 2048                 # tp_llama's --ctx; serve-tp.sh runs the default
QUEUE_RETRY_S = 5
QUEUE_RETRIES = 12         # a minute of "queue full" before giving up
UA = "llm-kv260-chat/1.0"


def request(url, body=None, timeout=5):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url, timeout=5):
    with request(url, timeout=timeout) as r:
        return json.load(r)


def explain(err, base):
    """One line for a failed request."""
    if isinstance(err, urllib.error.HTTPError):
        body = err.read().decode(errors="replace")
        try:
            msg = json.loads(body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = body.strip()[:300] or err.reason
        if err.code == 429:
            return f"HTTP 429: {msg}; try again in a few seconds"
        return f"HTTP {err.code}: {msg}"
    reason = err.reason if isinstance(err, urllib.error.URLError) else err
    if isinstance(reason, TimeoutError):
        return f"{base} did not answer in time; the boards may be stuck, bash board/tp/recover.sh brings them back"
    return f"cannot reach {base} ({reason}); the boards may be down, bash board/tp/recover.sh brings them back"


def open_chat(base, body):
    """POST the request; on 429 wait and retry, since the queue drains at one stream a time."""
    for attempt in range(QUEUE_RETRIES):
        try:
            return request(base + "/v1/chat/completions", body, timeout=600)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            e.read()
            print(f"queue full (one stream at a time, 8 may wait); retrying in {QUEUE_RETRY_S} s "
                  f"({attempt + 1}/{QUEUE_RETRIES})", file=sys.stderr)
            time.sleep(QUEUE_RETRY_S)
    raise RuntimeError(f"queue still full after {QUEUE_RETRIES * QUEUE_RETRY_S} s")


def note_queue(base):
    """Say so before sending if the prompt will wait behind another stream."""
    try:
        h = get_json(base + "/health")
    except (urllib.error.URLError, OSError, ValueError):
        return
    if h.get("queued", 0) > 0:
        print(f"[{h['queued']} waiting, one stream at a time; this prompt is queued]", file=sys.stderr)
    elif h.get("busy"):
        print("[the boards are on another prompt; this one is queued]", file=sys.stderr)


def chat(base, messages, max_tokens, temperature, stream):
    """Send the conversation and print the reply as it arrives.

    Returns (text, info): info has n, first_s, wall_s, timings, finish,
    stopped. On Ctrl-C the text is what arrived so far and stopped is True.
    """
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": stream}
    info = {"n": 0, "first_s": None, "wall_s": 0.0, "timings": None, "finish": None, "stopped": False}
    text = ""
    t0 = time.monotonic()
    try:
        with open_chat(base, body) as resp:
            if not stream:
                j = json.load(resp)
                choice = j["choices"][0]
                text = choice["message"]["content"]
                info["first_s"] = time.monotonic() - t0
                info["n"] = (j.get("usage") or {}).get("completion_tokens", 0)
                info["finish"] = choice.get("finish_reason")
                info["timings"] = j.get("timings")
                sys.stdout.write(text)
            else:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    j = json.loads(payload)
                    choice = j["choices"][0]
                    delta = choice.get("delta", {}).get("content", "")
                    if delta:
                        if info["first_s"] is None:
                            info["first_s"] = time.monotonic() - t0
                        text += delta
                        info["n"] += 1
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                    if choice.get("finish_reason"):
                        info["finish"] = choice["finish_reason"]
                    if "timings" in j:
                        info["timings"] = j["timings"]
    except KeyboardInterrupt:
        info["stopped"] = True
        sys.stdout.write("\n[stopped reading; the server runs this request to its end, nothing server-side "
                         "checks for a departed client, so the next prompt waits behind it]")
    info["wall_s"] = time.monotonic() - t0
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return text, info


def status_line(base, info):
    """tokens, rate, first token, the per-token split and the boards' power."""
    tm, parts = info["timings"], []
    if tm:
        parts.append(f"{tm['predicted_n']} tokens · {tm['predicted_per_second']:.1f} tok/s · "
                     f"first token {tm['prompt_ms'] / 1e3:.2f} s")
        # The server's number is the prefill alone; a longer wait on this side was the queue.
        if info["first_s"] is not None and info["first_s"] - tm["prompt_ms"] / 1e3 > 1.0:
            parts[-1] += f" (queued {info['first_s'] - tm['prompt_ms'] / 1e3:.0f} s)"
        if tm["prompt_n"] + tm["predicted_n"] > CTX - 250:
            parts.append(f"context {tm['prompt_n'] + tm['predicted_n']} / {CTX}, /reset before it runs out")
    else:
        n, first, wall = info["n"], info["first_s"], info["wall_s"]
        tps = n / (wall - first) if first is not None and wall > first and n > 1 else 0.0
        parts.append(f"{n} tokens · {tps:.1f} tok/s · first token {first if first is not None else wall:.2f} s (client-side)")
    if info["finish"] == "length":
        parts.append("cut at max_tokens")
    try:
        st = get_json(base + "/stats")
    except (urllib.error.URLError, OSError, ValueError):
        st = None
    if st:
        c = st.get("last_chat") or {}
        # Only the chat just finished: the split is for whatever ran last.
        if c.get("n_gen") and (not tm or c["n_gen"] == tm["predicted_n"]):
            parts.append(f"engine {c['engine_ms']:.1f} ms + cpu {c['cpu_ms']:.1f} + exchange {c['exchange_ms']:.1f} per token")
        pw = [b.get("power_w", 0.0) for b in st.get("boards", [])]
        if any(p > 0 for p in pw):
            parts.append("boards " + " + ".join(f"{p:.1f}" for p in pw) + f" = {sum(pw):.1f} W")
    print("[" + " · ".join(parts) + "]")


def show_stats(base):
    try:
        st = get_json(base + "/stats")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(explain(e, base))
        return
    print(f"{'rank':<5}{'host':<10}{'state':<10}{'temp':>7}{'power':>8}{'cpu':>6}{'ram':>14}  "
          f"{'per token: engine + cpu + exchange':<36}{'t/s':>6}")
    for b in st.get("boards", []):
        print(f"{b['rank']:<5}{b['host']:<10}{b['state']:<10}{b['temp_c']:>6.1f}C{b['power_w']:>6.1f} W"
              f"{b['cpu_pct']:>5.0f}%{b['mem_used_mb'] / 1024:>7.2f}/{b['mem_total_mb'] / 1024:.1f} GB  "
              f"{b['engine_ms']:>5.1f} + {b['cpu_ms']:.1f} + {b['exchange_ms']:.1f} ms{'':<15}{b['tps']:>6.1f}")
    c = st.get("last_chat") or {}
    if c.get("n_gen"):
        print(f"last chat: {c['n_gen']} tokens at {c['tps']:.1f} tok/s, {c['n_prompt']}-token prompt in "
              f"{c['prompt_ms'] / 1e3:.2f} s; per token engine {c['engine_ms']:.1f} + cpu {c['cpu_ms']:.1f} + "
              f"exchange {c['exchange_ms']:.1f} ms (mean over the boards)")
    print(f"{'decoding' if st.get('busy') else 'idle'}, {st.get('queued', 0)} waiting; "
          f"{st.get('tokens_total', 0)} tokens served since start; model {st.get('model', '?')}")


def ask(base, messages, prompt, args):
    """One turn: send, print, status line. Returns False if the request failed."""
    messages.append({"role": "user", "content": prompt})
    note_queue(base)
    try:
        text, info = chat(base, messages, args.max_tokens, args.temperature, not args.no_stream)
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as e:
        print(explain(e, base), file=sys.stderr)
        messages.pop()
        return False
    if text:
        messages.append({"role": "assistant", "content": text})
    else:
        messages.pop()
    if not info["stopped"]:
        status_line(base, info)
    return True


def main():
    ap = argparse.ArgumentParser(description="terminal chat with the four boards")
    ap.add_argument("prompt", nargs="?", help="answer this once and exit; without it, a REPL")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"the API (default {DEFAULT_URL})")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--system", metavar="TEXT", help="a system prompt ahead of the conversation")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 is greedy, the exact argmax (default)")
    ap.add_argument("--no-stream", action="store_true", help="print the reply whole instead of as it arrives")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    system = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(system)

    if args.prompt is not None:
        sys.exit(0 if ask(base, messages, args.prompt, args) else 1)

    try:
        h = get_json(base + "/health")
        print(f"Edge LLM on FPGAs: {base}, {h.get('ranks', '?')} boards"
              f"{', busy' if h.get('busy') else ''}. /reset, /stats, /quit.")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(explain(e, base), file=sys.stderr)
        print("(prompts will be tried anyway)", file=sys.stderr)
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/reset":
            messages[:] = list(system)
            print("conversation cleared")
            continue
        if line == "/stats":
            show_stats(base)
            continue
        if line.startswith("/"):
            print("commands: /reset, /stats, /quit")
            continue
        print()
        ask(base, messages, line, args)


if __name__ == "__main__":
    main()
