# 0008 — An OpenAI-compatible API on the four boards: 19–21 tokens/s

**Status: Attested.** 2026-09-14, `kv260-01..04`, `tp_llama --serve`.
Raw log: `evidence/0008-api-bench.txt`.

## The model

**Llama-3.2-1B-Instruct**, 1.24 B parameters, all tensors Q4_0: **663.16 MiB
on disk** (F16: 2.36 GiB). Sharded four ways, each board holds 173.8 MB of
weights in engine memory. The Instruct variant replaces the base model of
`0006`/`0007` because a chat endpoint on a base model answers nothing; same
architecture, same size, same engine, converted and quantized the same way
(`--output-tensor-type q4_0 --token-embedding-type q4_0`).

## Result

Client-side, streaming, from the PC (`board/tp/api-bench.py`), a 22-token
chat prompt, 67 tokens generated, greedy, three repeats:

| run | first token | tokens/s (client) | tokens/s (server) |
|---|---|---|---|
| 1 | 1.01 s | 18.95 | 18.95 |
| 2 | 0.93 s | 20.51 | 20.51 |
| 3 | 0.94 s | 20.86 | 20.86 |

Client and server agree to the second decimal: HTTP and SSE add nothing
measurable per token. The rate is the four-board tensor-parallel rate of
`0007` (21–24 t/s on a 6-token prompt) less a little for the longer
context in attention. First-token latency is the prompt at ~22 tokens/s —
prefill runs the same one-pass-per-token path as decode.

A non-streaming request for comparison: 45 tokens in 2.36 s = 19.0 t/s,
prompt 19 tokens in 1.07 s.

## What was built

`tp_llama --serve PORT` keeps all four ranks resident. Rank 0 runs
cpp-httplib (llama.cpp's bundled copy) with:

- `POST /v1/chat/completions` — messages through the model's own chat
  template (`llama_chat_apply_template`; the GGUF carries Llama 3's), with
  `stream: true` as server-sent events in OpenAI's chunk format;
- `POST /v1/completions` — raw prompt;
- `GET /v1/models`, `GET /health`;
- `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`; responses carry
  `usage` and llama-server-style `timings`.

A request is tokenized on rank 0 and broadcast over the TCP mesh; the four
ranks then run the same lockstep loop as the one-shot mode. Sampling is
replicated rather than centralised: every rank contributes the top-k
logits of its vocab shard (40 × 8 bytes), and every rank runs the same
seeded sampler over the union, so the chosen token is identical everywhere
with no extra round trip. The mesh is one pipeline, so one request at a time.

## Running it

```bash
board/tp/serve-tp.sh 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114      # start (detached on the boards)
curl http://10.42.0.137:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In one sentence, what is an FPGA?"}],"max_tokens":64}'
.venv/bin/python board/tp/api-bench.py http://10.42.0.137:8080            # tokens/s as a client sees it
board/tp/serve-tp.sh stop 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
```

Any OpenAI client works with `base_url="http://10.42.0.137:8080/v1"` and
any API key.

## Limits

- **Prefill is one engine pass per token**, so a 500-token prompt is ~25 s
  to first token. A batched engine (`0006` next steps) is the fix.
- Sampling is over the union of per-rank top-40 candidates; with top-p
  that is exact in practice, but it is not the full 128k-way softmax.
- Context is 2048 tokens per request and the KV cache is not kept between
  requests.
- One `ssh` trap on the way: a rank detached with `nohup … &` alone keeps
  the ssh session open; `setsid nohup … < /dev/null &` does not.
