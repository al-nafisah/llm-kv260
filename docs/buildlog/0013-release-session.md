# 0013 — One recorded session: 1, 2 and 4 boards, module power, and the CPU baseline

**Status: Attested.** 2026-11-04, `kv260-01..04`, the shipped gemv2 bitstream
(md5 `81e7b63d430956c99de998ab90bfffda` on every board), the runtime and
model of `0012`. One script ran everything back to back from the PC,
`board/tp/bench/session.py`, while it sampled every K26 module's INA260
about ten times a second; `board/tp/bench/session-report.py` turned the
raw files into `evidence/0013-session-report.txt`. Raw logs:
`evidence/0013-*`.

## Result

**Rate and module power, 400 tokens.** `run-tp.sh`, the `0012` prompt
(24 tokens, Instruct model, greedy), 400 generated tokens, so positions
24–423; three runs per board count, in this order
(`evidence/0013-tp-runs.txt`, `evidence/0013-session-report.txt`):

| boards | tokens/s (3 runs) | per token: engine | exchange | CPU | module power, boards running | J per token |
|---|---|---|---|---|---|---|
| 1 | 13.39, 13.51, 13.53 | 53.4 ms | 0.2 | 22.0–22.9 | 6.16–6.25 W | 0.459–0.462 |
| 2 | 21.87, 21.75, 21.65 | 22.4–22.5 | 12.4–12.9 | 11.8–11.9 | 11.59–11.68 W | 0.530–0.539 |
| 4 | 34.71, 36.69, 36.57 | 10.0 | 11.4–12.7 | 5.9–6.8 | 22.39–22.76 W | 0.620–0.645 |

The per-token columns are the runtime's buckets, every rank of the three
runs: each sums all 423 forwards of a run (the 24 prompt forwards included)
and divides by the 400 generated tokens, so they read about 6% above the
generation-only split that `/stats` gives below.

Two boards decode 1.61 times as fast as one and four boards 2.71 times
(medians 13.51, 21.75, 36.57 tokens/s), a parallel efficiency of 80% and
68%. Energy per token rises with the board count: each added board brings
its own idle 3.84–3.94 W, while the rate grows less than in proportion. The
four-board runs repeat `0012`'s 36.21–36.37 except the first, which spent
12.7 ms per token in the exchange instead of 11.4–11.7; the first API reply
of each kind shows the same (Traps).

**Idle.** 15.54–15.58 W for the four modules, 3.84–3.94 W each, the same
with the runtime resident and the API waiting (15.563 W), with no runtime
(15.577 W) and at the end (15.543 W). Decoding on four boards therefore
adds 6.8–7.2 W to the idle figure.

**Through the API**, from the PC over the LAN, streamed, temperature 0
(`evidence/0013-api-replies.txt`):

| reply | tokens/s, server (3 runs) | first token | module power, decoding | J per token |
|---|---|---|---|---|
| 400 tokens, 33-token chat prompt | 33.77, 35.87, 36.14 | 0.75–0.84 s | 22.36–22.70 W | 0.627–0.662 |
| 67 tokens (stops at the end of the answer), 22-token prompt | 38.02, 41.97, 41.41 | 0.49–0.51 s | 23.04–23.63 W | 0.560–0.606 |

The client's own rate (tokens 2..n over first to last streamed token)
agrees with the server's within 0.09 tokens/s. In the 400-token replies
`/stats` splits a token into engine 9.42–9.48 ms, exchange 10.9–12.5 ms
and CPU 6.1–7.2 ms per rank; the short replies run at shallower context,
where attention is cheaper (CPU 2.8–3.9 ms).

**The engine, each board alone** (`evidence/0013-bench-gemv2.txt`):
`EXACTNESS: PASS` on all four boards, 9 of 9 contract cases, 0 rows
differing. One token's 65 calls back to back: MAC time 13.30–13.41 ms for
173.8 MB, 12.96–13.06 GB/s; with 300 µs gaps (as in the stream, where the
exchange leaves room to prefetch) 8.05–8.14 ms. While a board's bench runs,
its module peaks at 6.88–7.07 W.

**The CPU baseline**, the same board and model file without the engine
(`evidence/0013-cpu-llama-bench.txt`): llama.cpp's `llama-bench` on
`kv260-01`'s four A53 cores, `-p 0 -n 400 -r 3 -t 4 -ngl 0 -dev none`,
runtime stopped on all four boards, decodes **2.88 ± 0.07 tokens/s**
(mean ± standard deviation of three 400-token runs from an empty context,
so positions 0–399). The module drew 5.28 W (mean over the phase after its
first 30 s, 3,731 samples), 1.83 J per token. On the same board the engine
path decodes 13.39–13.53 tokens/s at 6.16–6.25 W, 0.46 J per token. The two
are different software (llama.cpp's NEON kernels on four threads against
this runtime on one core and the engine), and llama-bench decodes from an
empty context without sampling text, so the ratio (4.7 times the rate,
about a quarter of the energy per token) describes this board with and
without its fabric, not llama.cpp in general. The boards' llama.cpp is a
source tarball without a recorded revision (`build: unknown (0)`).

**The text.** Each configuration gives the same 400 tokens on every run
(md5 per board count in `evidence/0013-text.txt`), and the four-board text
is byte-identical to `0012`'s. The 1-, 2- and 4-board texts differ from
each other: the two-board text leaves the four-board one at character 72,
the one-board text at character 673. Row-parallel matrices (attention
output, FFN down) add the ranks' partial sums, so the float32 additions
group differently for each board count and a near-tie in the greedy argmax
can go the other way. The engine is bit-exact to the contract on every call
the bench tests; the contract defines one matrix call, and how a split
matrix's partial sums are added is the runtime's.

**The demo.** The README's GIF replays one chat, recorded through the API on
the night before this session (2026-11-04 00:40 local time,
`evidence/0013-demo-take-3.txt`): a 32-token chat prompt
processed in 0.85 s, then 143 tokens to the end of the answer at 34.62
tokens/s (server-side). Its power figures are `/stats` samples, which the
runtime takes once per generated token.

## What this does and does not establish

- The decode rate and the module power now come from the same runs. Four
  boards: 34.7–36.7 tokens/s at 22.4–22.8 W, 0.62–0.65 J per generated
  token. One board: 13.4–13.5 tokens/s at 6.2 W, 0.46 J. This supersedes
  `0012`'s one-board 16.85 tokens/s (64 tokens at positions 12–75, first
  gemv2 bitstream) as the scaling baseline: at positions 24–423 one board
  spends 22 ms per token on the CPU, attention over all 32 heads included.
- Power is the **K26 module only** (its INA260): the carrier cards, fans,
  power supplies and the switch are not measured, and no wall-power meter
  was used. Idle is measured, so the decode increment is known; the idle
  share is large (15.54–15.58 of 22.39–22.76 W).
- The INA260 is read about ten times a second (its readings come in 10 mW
  steps); a decode window holds 66–72 samples per board for four boards and 243–249 for
  one. run-tp.sh only returns when every rank has exited, so its decode
  window is taken as [end − generation time + 1.5 s, end − 2.5 s], inside
  generation for any teardown between 0 and 2.5 s; the API windows are the
  first and last streamed token on the PC clock.
- The earlier power record, one sample per token from `/stats` during a
  400-token chat on 2026-09-15 (22.57 W at 35.53 tokens/s), agrees and is
  kept as `evidence/0013-decode-power-2026-09-15.txt`. Its "idle" rows are
  not idle power: `/stats` samples the sensor once per generated token, so
  between chats it repeats the last decoding sample.
- The CPU baseline agrees with `0006`'s 2.98 tokens/s (tg64, the base
  model, `evidence/0006-llama-bench-kv260-01.txt`, restored to the tree with
  the other three `0006` logs the history rebuild had left out).
- Nothing here is compared with another system's numbers.

## Configuration

Boards as in `0012`: Ubuntu 22.04, 5.15.0-1027-xilinx-zynqmp, MTU 9000,
DisplayPort interrupts 0 per 2 s, UIO devices `gemv_fetch` and `gemv_mac`,
the runtime on CPU 2, the sampler (the shell's `read` builtin and
`sleep 0.1`) on CPU 1 (`evidence/0013-env.txt`). Model: the
Llama-3.2-1B-Instruct GGUF with every weight tensor in Q4_0, 703,209,184
bytes (sha256 `b940c6c6…3f1b13`); a token streams 695,107,584 bytes of
weights, 173,776,896 per board on four. The session's phases with their
PC-clock start and end are in `evidence/0013-phases.txt`; the raw power
samples, PC time and watts, are `evidence/0013-power-kv260-0N.txt`.

## Traps

- **`llama-bench` on the boards' llama.cpp failed first**: that build
  carries the old engine's backend, which registers a KV260 device even
  when its overlay is not loaded, and context creation fails with
  `-ngl 0` alone. `-dev none` keeps it off (`evidence/0013-cpu-llama-bench.txt`
  keeps the failed attempt too).
- **The first four-board run after a pause is slower**, in the exchange:
  12.1–12.7 ms per token (the first run-tp run, the first API reply of each
  length) against 10.5–11.7 for the runs after it. The two-board runs do
  not show it. Report every run, not the best.
