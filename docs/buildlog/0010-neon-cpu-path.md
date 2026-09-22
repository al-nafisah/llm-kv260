# 0010 — The CPU term: 32 ms → 6 ms per token at 400 tokens of context

**Status: Attested.** 2026-09-14, `kv260-01..04`, four-board stream, from the
scalar build (before) to the NEON build measured here (after); both are rows
of the table below. Raw logs: `evidence/0010-*`. Runtime:
`board/tp/tp_llama.cpp`; microbenchmarks in `board/tp/bench/`.

## Result

The public page showed a 352-token chat at 15.1 t/s with 28 ms of every
65 ms token "in cpu", against the 8.7 ms that `0007` measured on 64-token
runs. The CPU term was growing with the conversation: three samples at
different depths (8.7 ms at ~position 38, 27.9 ms averaged over a chat
ending at 373, 48.0 ms on that chat's last token, `evidence/0007-tp-scaling.txt`
and the live `/stats`) lie on one line, **0.12 ms per context position per
token plus 4 ms**. The only per-position work on the CPU is attention over
the KV cache, and a microbenchmark of that loop alone reproduces the line
(46.8 ms at position 400, `evidence/0010-microbench.txt`, `attn_bench`).

Same prompt (27 tokens), 400 tokens generated, the four-board stream, the
Instruct model, greedy (`evidence/0010-tp-neon-runs.txt`):

| build | tokens/s | per token: engine | exchange | CPU |
|---|---|---|---|---|
| before (scalar ops) | 14.47 | 19.3–19.6 ms | 19.1–19.9 ms | **31.6–32.2 ms** |
| NEON ops, interleaved KV cache, no unroll pragmas | 18.79 | 19.1–20.2 | 18.6–21.1 | 14.7–16.1 |
| NEON ops, per-head KV cache, pragmas (committed), 3 runs | **23.03, 22.69, 23.05** | 19.0–22.1 | 16.5–20.7 | **5.9–7.0 ms** |

Through the API, three repeats of the 22-token prompt with 67 tokens
generated: 28.0 / 24.7 / 24.7 t/s (`evidence/0010-api-bench.txt`; `0009`
measured 20.8–21.9 on the same prompt). A 400-token chat through the API:
23.1 t/s, CPU 6.5 ms of a 43 ms token (`evidence/0010-stats-last-chat.txt`).

## Cause

Three things, each measured in `evidence/0010-microbench.txt`:

1. **The ops were scalar.** GCC 11 at `-O2` (the board's compiler) does not
   vectorize float reductions, so every dot product in attention was a
   serial chain of 64 dependent `fmadd`s on an in-order core, and each K
   and V row was read once per query head. Writing attention in NEON
   intrinsics with the 4 query heads of a KV group sharing one pass over
   K and one over V gives 4.5x on the loop alone (`attn_bench`, 46.8 →
   10.6 ms at position 400). The constant part is the same story:
   rmsnorm 1.3 → 0.3 ms, SiLU·up 2.3 → 0.5 ms, residual adds 0.5 → 0.1 ms
   (`misc_bench`, `ops_bench`).
2. **The kernel's fixed-count loops stayed rolled.** The first build of the
   NEON kernel ran 2.3x slower in the runtime than in the microbenchmark:
   at `-O2` GCC 11 leaves a `for (b < B)` loop over a template constant
   rolled, so the accumulator arrays live on the stack. `#pragma GCC
   unroll 4` on each of them fixes it.
3. **The KV cache layout.** Rows were `[layer][pos][kv_rows]`: one KV head's
   256 bytes at a 512-byte stride, half of every line pair unused, and the
   A53 stalls on each miss. Sweeping those lines with no arithmetic at all
   costs 5.1 ms at position 400; the same data laid out `[layer][kv
   head][pos][d]` costs 3.0 ms, and the kernel on it drops from 15.3 to
   8.7 ms (`attn_var`). Software prefetch and wider column chunks were
   each worth 10–20% on the strided layout and nothing on the contiguous
   one, so neither is used.

Two worker threads (one per KV head) would take the attention term from
~7 to ~6 ms at position 400 and 18 → 13.5 at 1000 (`attn_bench`); not
taken, to keep the runtime single-threaded.

## What this does and does not establish

- The CPU term is now ~5 ms at the start of a conversation and grows by
  about 0.02 ms per context position instead of 0.12; at 400 tokens of
  context a token is engine 19 + exchange 19 + CPU 6 ms.
- The generated text changed: attention now sums in a different order
  (four lanes, then a tree) and softmax uses ggml's vector `expf`, so the
  logits differ in the last bits and greedy decoding diverged from the old
  text after 80 tokens (`evidence/0010-tp-neon-text.txt`; both essays are
  coherent). Every rank still prints the same text, and the three
  committed-build runs produced identical text — the CPU path is
  deterministic, it is just not the old path. The bit-exact contract
  (engine vs `q4_0.py`) is untouched: no GEMV input or output changed.
- The 1.4 ms of `quantize_row_q8_0_ref` per token (65 calls) is still
  scalar and still counted in the *engine* bucket; it is left alone
  because its rounding (`roundf`) is what the activation contract expects
  and ggml's NEON quantizer rounds to nearest-even.
- The exchange (16–21 ms) is now the largest term again, as `0007` said.

## Configuration

Board: Ubuntu 22.04, gcc 11.4.0, `-O2 -std=c++17`, governor `userspace` at
1.333 GHz on all cores (checked: no frequency scaling was involved).
Runtime built on each board with `make` in `~/board/tp`; the four ranks'
binaries were rebuilt from the same source (md5 checked) before each
"after" run. Microbenchmarks: `board/tp/bench/`, `make run` on a board
(`ops.inc` is the ops section of `tp_llama.cpp`, extracted by the
Makefile, so `ops_bench` and `attn_var` time the committed kernels).
Model: `llama-3.2-1b-instruct-q4_0-pure.gguf`, `--ctx 2048`.

## Traps

- **A per-token number that grows with the conversation looks like a
  regression in the wrong place.** `0007` measured 8.7 ms of CPU on
  64-token runs and the page showed 28 ms; nothing had changed. Any
  per-token figure on this system has to say at what context depth it was
  taken, and a 64-token run cannot see this term.
- **The microbenchmark and the runtime disagreed by 2.3x** on the same
  source text, because of how the template was compiled inside the larger
  file (rolled loops, spilled accumulators). Benchmark the extracted
  kernel from the committed file (`ops.inc`), not a copy.
- The board microbenchmarks vary by ±20% run to run (scalar attention at
  position 400 measured 46.8 and 55.0 ms in two runs); the runtime's
  per-token split over 400 tokens is the number to trust.
