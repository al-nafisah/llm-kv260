#!/usr/bin/env python3
"""Measure tokens/s through the OpenAI-compatible API, the way a client sees it.

    .venv/bin/python board/tp/api-bench.py http://10.42.0.137:8080 [--n 128] [--repeats 3]

Streams a chat completion, times first token and the generation phase from
the client side, and prints the server's own timings next to them.
"""

import argparse
import json
import sys
import time
import urllib.request


def one(base, prompt, n, stream=True):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n, "temperature": 0, "stream": stream}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "llm-kv260-bench/1.0"})
    t0 = time.time()
    text, first, n_tok, timings, usage = "", None, 0, None, None
    with urllib.request.urlopen(req, timeout=600) as resp:
        if not stream:
            j = json.load(resp)
            text = j["choices"][0]["message"]["content"]
            return text, time.time() - t0, None, j.get("usage"), j.get("timings")
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            delta = j["choices"][0].get("delta", {}).get("content", "")
            if delta:
                if first is None:
                    first = time.time() - t0
                text += delta
                n_tok += 1
            timings = j.get("timings", timings)
            usage = j.get("usage", usage)
    return text, time.time() - t0, first, usage, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt", default="Explain in a few sentences why the sky is blue.")
    a = ap.parse_args()
    for i in range(a.repeats):
        text, wall, first, usage, tm = one(a.base, a.prompt, a.n)
        gen = usage["completion_tokens"] if usage else 0
        client_tps = gen / (wall - first) if first and gen > 1 else float("nan")
        print(f"run {i+1}: {gen} tokens, first token {first:.2f}s, client-side {client_tps:.2f} t/s, "
              f"server-side {tm['predicted_per_second']:.2f} t/s (prompt {tm['prompt_n']} tok in {tm['prompt_ms']/1e3:.2f}s)")
        if i == 0:
            print("   ", text.replace("\n", " ")[:300])


if __name__ == "__main__":
    main()
