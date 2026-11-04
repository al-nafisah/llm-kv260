# 0012 — The fabric term: 18.8 → 10.0 ms per token; four boards 26.9 → 36.3 tokens/s

**Status: Attested.** 2026-09-14, `kv260-01..04`, four-board stream, from
the `0011` build to the build measured here. Raw logs: `evidence/0012-*`.
Runtime: `board/tp/tp_llama.cpp` (memory and engine sections); driver
`board/kv260gemv/gemv_dev.c`. The `bench_calls` and `bench_banks` benches
used below left the tree with the `gemv_q4_0` path they measured.

## Result

Same prompt (24 tokens), 400 tokens generated, four boards, Instruct model,
greedy; the first two rows are the `0011` binary and this one run back to
back (`evidence/0012-tp-ab-runs.txt`), the last is below
(`evidence/0012-tp-gemv2-runs.txt`):

| build | tokens/s | per token: engine | exchange | CPU |
|---|---|---|---|---|
| `0011` | 26.87 | 18.8–18.9 ms | 13.1–13.4 | 5.8–6.1 |
| staggered quarters + NEON quantize, old engine | 29.23 | 15.8–16.2 | 12.9–13.2 | 5.8–5.9 |
| gemv2 on every board, MTU 9000 and no display storm on all four, 3 runs | **36.37, 36.21, 36.36** | **9.9–10.0** | 11.7–12.0 | 5.9–6.1 |

The text is byte-identical between the two builds over the 400 tokens, and
on a single board over 64 tokens (`tp_llama.0011` vs `tp_llama`, same
file): the engine, the contract and the Q8 activation values are
untouched. The first step is two changes in software:

1. **The four engine streams were colliding in the DRAM banks.** The
   driver put engine m's quarter of a matrix at `w + m × quarter_bytes`,
   and for every Llama shape that stride is a multiple of 64 KB, so the
   four streams always sat at the same offset modulo 64 KB. Placing the
   quarters at distinct offsets (ports 1 and 3 shifted by 16 KB) takes
   gate|up from 10.3 to 12.7 GB/s and down from 10.3 to 12.7
   (`evidence/0012-bank-stagger.txt`: any pattern of distinct offsets
   gives 12.3–12.8; a 64 KB pad brings the collision back). The runtime
   now allocates each quarter separately (64 KB-aligned, 0 or 16 KB in)
   and `gemv_dev_start4` takes four addresses.
2. **The activation quantization was scalar.** ggml's
   `quantize_row_q8_0_ref` cost 30 µs of a 77 µs q|k|v call
   (`evidence/0012-bench-calls.txt`). The NEON version in `engine_t`
   produces the same bytes (`vcvta` rounds ties away from zero as `roundf`
   does; ggml's own NEON path uses ties-to-even and would not) in 8 µs.

Removing the invalidate *before* each call, which looked redundant, made
the output garbage from the first token: `rope()` writes q in place in the
engine's result buffer, so dirty lines exist and an eviction during the
next call lands on the engine's output. It is back, with a comment.

## What the fabric can and cannot do, measured

`bench_calls` on kv260-01 (`evidence/0012-bench-calls.txt`): 65 shard
calls back to back, one token's engine work with nothing between them,
take 16.45 ms; the runtime measured 17.8 ms around them, so the software
between calls was ~1.3 ms, and the rest is the stream. A call costs 2.8 µs
fixed plus ~2 µs of activation load at K = 2048.

**The clock is not the limit.** The same engine built at 300 MHz
(`GEMV_CLK_NS=3.333`, `GEMV_MHZ=300`; the PS delivered 333 MHz, and the
design missed timing by 0.31 ns yet stayed bit-exact on the vectors)
streams 12.0 GB/s on every shape, staggered or not
(`evidence/0012-clock-333.txt`). At 250 MHz with the stagger the same
shapes reach 12.7. The ceiling is on the DDR controller's side, at roughly
two thirds of the 19.2 GB/s peak, and it is not reached by clocking the
ports faster. The 333 MHz build is not used.

**Bytes cannot move on chip.** A board's shard is 174 MB; the K26 holds
0.65 MB of block RAM and 2.3 MB of UltraRAM. What the UltraRAM *can* do is
hold the head of the next matrix while the CPU is busy with attention or
the exchange, which is what `hls/gemv2/` does — see below.

## gemv2: the next matrix loads while the CPU works

The engine is idle for the 13 ms of exchange and 6 ms of CPU in every
token. `hls/gemv2/` splits the kernel so that the DRAM stream can run
through those gaps: `gemv_fetch` (four burst readers) streams a matrix's
bytes into eight AXI-Stream FIFOs in UltraRAM (16384 × 128 bit each: 2 MB,
all 64 URAM of the device) as soon as the runtime names it, and `gemv_mac`
(eight MAC units, the same arithmetic through `gemv_core.h`) consumes them
when the activations are ready. Two MAC units per port because a full FIFO
only helps if it drains faster than DRAM refills it: one unit per port
consumes 16 GB/s against a 12.7 GB/s DRAM, two consume 32. The layout that
keeps one sequential stream per port pairs row i with row i + rows/8 and
alternates their 144-byte chunks (`gemv2_chunk_offset`), copied chunk by
chunk from the GGUF at load. Design: [../fabric-engine.md](../fabric-engine.md).

Vivado, the build every board runs: 58,268 LUTs (50%), 76 BRAM tiles, 64
URAM (100%), WNS −0.210 ns on 577 endpoints (the first build: 58,832 LUTs,
−0.149 ns). Accepted on the same grounds as `0006`'s −0.049: the exactness
tests pass on hardware, on every board. Bit-exact on the contract's vectors on `kv260-01` and `kv260-02`,
including the four-rank shard shapes checked one call each
(`evidence/0012-gemv2-board1.txt`, `evidence/0012-gemv2-runs-1.txt`).

What the FIFOs buy, `bench_gemv2` on `kv260-01`
(`evidence/0012-gemv2-board1.txt`):

| shape (one rank of four) | cold: fetch and MAC started together | prefetched: fetch 1 ms earlier |
|---|---|---|
| q\|k\|v 0.89 MB | 66 µs, 13.3 GB/s | 31 µs, 28.6 GB/s |
| o 0.59 MB | 54 µs, 11.0 | 21 µs, 27.9 |
| gate\|up 4.72 MB | 348 µs, 13.6 | 197 µs, 23.9 |
| down 2.36 MB | 176 µs, 13.4 | 77 µs, 30.6 |
| lm_head 36.9 MB | 2.81 ms, 13.1 | 2.66 ms, 13.9 |
| a token's 65 calls, no gap | 13.29 ms | |
| a token's 65 calls, 300 µs gap after each | | **8.25 ms** of MAC time |

Cold, the pure reader streams 13.1–13.6 GB/s where the old engine's
staggered quarters did 12.7. In the runtime: one board alone, engine 66.8
→ 59.0 ms per token (its matrices are four times larger than the FIFO);
two boards, 19.78 → 21.50 t/s with the engine term 31.5 → 22.3 ms, text
identical to two boards on the old engine
(`evidence/0012-gemv2-board1.txt`, `evidence/0012-gemv2-runs-1.txt`).

Four boards with the drain fixed, `kv260-01` and `-02` on gemv2 and `-03`
and `-04` still on the old engine (their overlays need the user's sudo),
400 tokens, text identical to the reference above
(`evidence/0012-gemv2-runs-2.txt`):

| rank | engine | per token: engine | exchange | CPU |
|---|---|---|---|---|
| 0, 1 | gemv2 | **10.0 ms** | 17.7–17.8 (waiting for 2 and 3) | 5.9 |
| 2, 3 | gemv_q4_0, staggered | 16.3–16.4 | 11.7 | 5.7–5.8 |

The stream ran at 30.24 t/s, bounded by the two old-engine ranks; the
gemv2 ranks' engine term in the real flow, with the real gaps, is 10.0 ms
against 16.3. With every board on gemv2 the stream runs at 36.3 t/s (below).

**The drain trap.** A process that starts finds the FIFOs holding whatever
the previous process prefetched; `rows_per_mac = 0` drains them. The first
bitstream drained the eight FIFOs one after another, and when the previous
process had died with a fetch *stalled* half-delivered (a single-rank run
exits with the next token's layer-0 q|k|v, 3.5 MB, pending), the stalled
reader blocked on the FIFO not being read, the idle count ran out early,
and every later call read beats offset by the stale ones: garbage from the
first token of the mixed four-board run, and half the rows wrong on the
vectors, until one clean process passed (`evidence/0012-gemv2-runs-1.txt`).
The drain now reads all eight FIFOs in one loop.

## Every board on gemv2, and the exchange

The user loaded gemv2 on `kv260-03` and `-04` and ran the DisplayPort unbind
and the rewritten jumbo-frames script there. All four boards now have the
same engine, MTU 9000 and no display interrupts. Every ordered pair passes
9000-byte packets with do-not-fragment set, and gemv2 is bit-exact on all
nine vector cases on every board (`evidence/0012-board-settings.txt`,
`evidence/0012-gemv2-all-boards.txt`). Same prompt, 400 tokens, three runs,
text byte-identical to the reference (`evidence/0012-tp-gemv2-runs.txt`):

| run | tokens/s | per token: engine | exchange | CPU |
|---|---|---|---|---|
| 1 | 36.37 | 9.9–10.0 ms | 11.8 | 5.9 |
| 2 | 36.21 | 9.9–10.0 | 11.9–12.0 | 5.9 |
| 3 | 36.36 | 9.9–10.0 | 11.7–11.8 | 5.9–6.1 |

Through the API, the 22-token prompt with its 67-token reply: 39.98 / 40.09
/ 39.95 t/s at the client (`0011`: 29.4; `evidence/0012-api-bench.txt`). A
400-token chat through the API: 35.2 t/s, engine 9.4 ms, exchange 11.8, CPU
6.2 (`evidence/0012-stats-last-chat.txt`).

Boards 3 and 4 changed engine and network settings at the same time, and
the runtime's exchange bucket also holds the wait for the slowest rank, so
the runtime alone cannot say where the exchange's 12.9–13.2 → 11.7–12.0 ms
came from. `mesh_bench` has no engine in it. Pinned, against `0011`'s run
with the same committed code and mixed settings
(`evidence/0012-mesh-bench.txt`, `evidence/0011-mesh-bench.txt`, medians of
1000 per rank; this bench moves ±10% between sessions):

| four-way exchange, committed `comm_t` | `0011`: boards 1–2 at 9000, 3–4 at 1500 with the storm | now: all four at 9000, no storm |
|---|---|---|
| all-gather 64 B | 83.8–84.9 µs | 93.8–95.0 |
| all-gather 8192 B | mean 377.1–377.2 | mean 377.1 |
| all-gather 16384 B | median 638–644, max 3.4 ms | median 638–662, max 16.5 ms |
| **all-reduce of 2048 floats**, what the runtime sends | **363.7–364.6** | **282.0–282.5** |

The all-reduce the runtime issues 33 times a token is 82 µs cheaper. Its
messages are 2 KB per peer: one frame at MTU 9000, two at 1500. By the bench
that is ~2.7 ms a token; the runtime gained 1.2–1.4 ms, and the gap is not
explained here. The 64-byte floor rose by 10 µs, and 16 KB messages now
stall for ~16 ms now and then; the runtime sends neither size.

## What this does and does not establish

- The four-board stream is 36.2–36.4 t/s over 400 tokens with gemv2 on
  every board (three runs): engine 9.9–10.0 ms, exchange 11.7–12.0, CPU
  5.9–6.1, text byte-identical to `0011`'s. On the old engine with
  staggered quarters it was 29.2 t/s with an engine term of 15.8–16.2 ms,
  of which the weight stream at 12.7 GB/s is 13.7.
- The exchange is the largest term again. The bench credits the network
  settings with an 82 µs cheaper all-reduce; the runtime gained less than
  that predicts.
- The page's Memory BW figure divides the shard by the engine term. With
  gemv2 part of the stream runs during the exchange and the CPU work, so
  it reads ~18.5 GB/s per board, which is no longer a DRAM rate.
- Bank placement is now part of the contract between runtime and driver:
  a caller that lays out its own matrix must keep the four quarters at
  distinct offsets modulo 64 KB, or it gets `0011`'s speed.
- 300 MHz is not a lever on this board under Linux. Whether the DDR
  controller's QoS registers or two more ports (HPC0/1) would raise the
  12.7 is not measured.

## Configuration

Board: Ubuntu 22.04, 5.15.0-1027-xilinx-zynqmp, gcc 11.4 `-O2`; benches
pinned with `taskset -c 2`; the API server was idle but resident during
the microbenchmarks (they use `udmabuf2`, which the runtime does not
touch). PC: Vitis HLS and Vivado 2022.2; the build files took `GEMV_VARIANT`,
`GEMV_CLK_NS` and `GEMV_MHZ` so an experiment never overwrote the shipped
engine, and `hls/gemv2/run_hls.tcl` with `hw/kv260_gemv2/` still do.

gemv2: Vitis HLS 2022.2 at 4 ns (fetch estimated 3.600 ns, MAC 3.625),
Vivado 2022.2 at 250 MHz. Synthesis and implementation reports of the build
the boards run: `evidence/0012-gemv2-hls-fetch-csynth.txt`,
`evidence/0012-gemv2-hls-mac-csynth.txt`,
`evidence/0012-gemv2-vivado-timing-summary.txt`,
`evidence/0012-gemv2-vivado-utilization.txt`. The 333 MHz build of the old
engine: `evidence/0012-clock-333-vivado-timing-summary.txt`.

## Traps

- **A bench must not use `udmabuf0` while the runtime is resident**: both
  bump-allocate from the pool's start, and the bench would write random bytes
  over the loaded shard. `udmabuf2` is free.
- **Asking the PS for 300 MHz gives 333**: `PL0_REF_CTRL` divides a
  1.5 GHz PLL by integers; the overlay's `assigned-clock-rates` says 300
  and `clk_summary` shows 333333330.
- **The invalidate before an engine call is load-bearing** (above).
- **`jumbo-frames.sh` used to end in `netplan apply`**, and on boards whose
  only netplan file is cloud-init's it could not find the file it edited.
  It now writes the boot file and bounces the link from a detached process.
  A netplan eth0 without `dhcp4: true` would boot with no address. The file
  was checked first with `netplan generate --root-dir` into a temporary
  directory on the board, which needs no root.
- **`for x in $list` in the tool's zsh runs once**, with the whole string as
  one word: the first all-pairs ping loop printed nothing. Loops go in bash
  scripts.

## Corrections (2026-11-04)

- **Bank placement** (step 1, and the contract in What this does and does
  not establish). The driver's quarter stride is a multiple of 64 KB for the
  gate|up and down shards (18 and 9 × 64 KB) but not for q|k|v, o and
  lm_head (3.375, 2.25 and 140.9 × 64 KB; shard sizes in
  `evidence/0012-bench-calls.txt`). The runtime's placement gives two
  offsets modulo 64 KB, not four distinct ones: ports 0 and 2 at 0, ports 1
  and 3 at 16 KB (`board/tp/tp_llama.cpp:214-221`). In `bench_banks` only
  the two patterns that put all four quarters at the same offset (0,0,0,0
  and 0,4,8,12) collide, 10.24–10.30 GB/s on gate|up and down; every other
  pattern, two distinct offsets as well as four, reads 12.56–12.76 on those
  shapes (12.23–12.72 on q|k|v and lm_head, 10.00–10.18 on o), so "any
  pattern of distinct offsets gives 12.3–12.8" names the wrong condition and
  the wrong low end. In `bench_stagger`, spacing the quarters by 256 B to
  8 KB gives four distinct offsets and still reads 10.27–11.88 on gate|up
  and down (`evidence/0012-bank-stagger.txt`). A caller that lays out its
  own matrix should use the runtime's placement: all four quarters at one
  offset, or spaced by 256 B to 8 KB, measured slow, and placements with
  three quarters at one offset were not tested.
- **333 MHz.** It does not stream 12.0 GB/s on every shape, staggered or
  not: gate|up reads 11.71–11.88, down 11.44–11.76 and lm_head 11.96–11.98
  whatever the placement, while q|k|v (10.67–11.38) and o (8.07–10.32) still
  lose when the quarters collide; the full matrices read 11.02–11.99
  (`evidence/0012-clock-333.txt`). The conclusion stands: nothing at
  333 MHz beats the 12.7 at 250 MHz. Exactness at 333 MHz was checked on one
  case, `k8192_n16`, and Vivado timed the build at 300.03 MHz (WNS
  -0.307 ns, `evidence/0012-clock-333-vivado-timing-summary.txt`), not at
  the 333 MHz the board ran. "Roughly two thirds of the 19.2 GB/s peak"
  divides a measured rate by a cited one, which
  [../measurement.md](../measurement.md) rules out: the measured 12.7 GB/s
  and the 19.2 GB/s DDR4 interface peak (Cited: Li et al., arXiv
  2502.10659) stand side by side, and that the limit sits in the DDR
  controller is this log's inference.
- **PL0_REF_CTRL** (Traps): 1.5 GHz divided by 5 is exactly 300 MHz, so a
  1.5 GHz PLL does not explain the 333 MHz. The delivered 249,999,998 Hz
  (`evidence/0012-board-settings.txt:12`) and 333,333,330 Hz
  (`evidence/0012-clock-333.txt:1`) both match a source of about 1.0 GHz
  divided by 4 and by 3.
- **A call's fixed cost.** The x2048 case (4.6 µs) streams 18,432 bytes of
  weights against the tiny case's 2,304, so its 1.8 µs over the 2.8 µs is
  not activation load alone (`evidence/0012-bench-calls.txt:5-6`).
- **The FIFO table and the runs after it** are the first gemv2 bitstream
  (`evidence/0012-gemv2-board1.txt:1`). The shipped, drain-fixed bitstream
  on all four boards (`evidence/0012-gemv2-all-boards.txt`) reads cold
  13.30–13.58 GB/s on q|k|v, gate|up and down, 13.10–13.11 on lm_head and
  10.96–11.01 on the o shard (the "13.1–13.6" above leaves o out);
  prefetched 27.86–30.62 on q|k|v, o and down, 23.95–24.07 on gate|up and
  13.84–14.00 on lm_head; a token's 65 calls take 13.31–13.46 ms of MAC
  time back to back and 8.01–8.15 ms with 300 µs gaps. One board alone
  holds the whole model, four times the four-board shards, which is 1.1
  (o) to 70 (lm_head) times the 2 MiB of FIFO rather than four times. The
  two-board step 19.78 → 21.50 t/s compares kv260-03 and -04 (old engine,
  MTU 1500, DisplayPort storm) with kv260-01 and -02 (gemv2, MTU 9000, no
  storm): engine, boards and network changed together
  (`evidence/0012-gemv2-runs-1.txt:17-22`).
- **The exchange gain.** The runtime issues 32 all-reduces a token (the
  33rd exchange is the argmax all-gather), so the bench predicts
  32 × 82.3 µs = 2.6 ms. The runtime's exchange went from 12.9–13.2 ms in
  the staggered run to 11.8 in gemv2 run 1 on every rank, 1.1–1.4 ms per
  rank, and 1.0–1.5 ms over the three gemv2 runs
  (`evidence/0012-tp-ab-runs.txt:17-20`, `evidence/0012-tp-gemv2-runs.txt`),
  not 1.2–1.4; boards 3 and 4 also changed engine between the two.
- **The run-tp.sh buckets.** The per-token engine, exchange and CPU figures
  that `run-tp.sh` prints (every table here except the API chat) are totals
  over the 24 prompt forwards and the 399 generation forwards divided by the
  400 generated tokens (`board/tp/tp_llama.cpp:945-963, 1322-1324`). They
  add up to more than the wall time per token: run 1's 10.0 + 11.8 + 5.9 =
  27.7 ms against 27.5 ms at 36.37 t/s. `/stats` splits the generation
  alone: engine 9.4, exchange 11.8 and CPU 6.2 ms in the API chat, with
  1.1 ms of each 28.5 ms forward untimed
  (`evidence/0012-stats-last-chat.txt`).
