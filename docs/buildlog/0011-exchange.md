# 0011 — The exchange: 19 ms → 13 ms per token, 23 → 27 tokens/s

**Status: Attested.** 2026-09-14, `kv260-01..04`, four-board stream, from
the `0010` build to the build measured here. Raw logs: `evidence/0011-*`.
Runtime: `board/tp/tp_llama.cpp` (comm section); exchange benchmark
`board/tp/bench/mesh_bench.cpp`, launched by `board/tp/bench/run-mesh.sh`.

## Result

Same prompt (27 tokens), 400 tokens generated, four boards, Instruct model,
greedy (`evidence/0011-tp-exchange-runs.txt`):

| build | tokens/s | per token: engine | exchange | CPU |
|---|---|---|---|---|
| `0010` (all-gather all-reduce, blocking reads, scheduler placement) | 23.03, 22.69, 23.05 | 19.0–22.1 ms | 16.5–20.7 ms | 5.9–7.0 ms |
| `0011` (reduce-scatter + all-gather, polled reads, pinned to CPU 2), 3 runs | **27.02, 27.01, 26.99** | 18.9–19.0 | **13.1–13.3** | 5.8–6.1 |

The text is byte-identical to the `0010` runs: the new all-reduce produces
the same float32 sums bit for bit. Through the API, three repeats of the
22-token prompt with 67 tokens: 29.4 / 29.4 / 29.4 t/s (`0010`: 24.7–28.0;
`evidence/0011-api-bench.txt`). A 400-token chat through the API: 26.6 t/s,
exchange 12.7 ms of a 37 ms token (`evidence/0011-stats-last-chat.txt`).

## What the exchange costs, measured alone

`mesh_bench` runs the runtime's own `comm_t` (extracted from the source by
the bench Makefile) in a loop with nothing between exchanges, so the cost
is the exchange without engine time or rank skew
(`evidence/0011-mesh-bench.txt`, medians of 1000):

| four-way all-gather of | scheduler placement | pinned to CPU 2 |
|---|---|---|
| 64 B | 113 µs | 82 µs (polled reads: 69) |
| 2048 B | 250 | 170 |
| 4096 B | 319 | 227 |
| 8192 B (the runtime's message) | 408 (mean 431, p90 537) | 377 (p90 411) |
| 16384 B | 662 | 649 |
| all-reduce of 2048 floats, `0010` code | 426 | 412 |
| the same as reduce-scatter + all-gather, polled | 341 | 339 |

Three findings:

1. **The wire is the cost.** Above 2 KB the time grows ~30 µs per KB of
   message: a rank sends 24 KB and receives 24 KB per 8 KB all-gather, and
   at 1 Gbit/s that is ~0.2 ms of the 0.4. The fixed part (a 64-byte
   four-way round) is 82 µs. So the bytes had to go down. The all-reduce is
   now a **reduce-scatter and an all-gather**: each rank receives the
   others' partials for its own quarter of the vector (3 × 2 KB), sums them
   in rank order, and sends the summed quarter to everyone (3 × 2 KB) —
   half the bytes in two rounds, and the sums are the same float32 additions
   in the same order as before, which is why the text did not change.
   Float16 partials would halve the bytes again at the price of the
   hidden state's precision; not taken (user decision).
2. **Where the process runs matters.** The scheduler put the ranks on CPU 0
   or 3. CPU 0 takes ~10,500 DisplayPort interrupts per second on every
   board (628 million since boot, no monitor attached,
   `evidence/0011-board-facts.txt`); CPU 3 takes the NIC's IRQ. Pinned to
   CPU 2, the floor drops 113 → 82 µs and the 8 KB p90 537 → 411 µs. The
   runtime now pins itself (`--cpu`, default 2).
3. **Polled reads** (non-blocking `recv` round-robin instead of a blocking
   `read` per peer) are worth ~13 µs on the floor and a shorter tail; the
   rank keeps its core instead of sleeping and being woken once per peer.

In the runtime the exchange bucket also absorbs rank skew (the wait for the
slowest engine), which is why 33 × 0.34 ms ≈ 11 ms measures as 13.

## Board settings

Applied and persisted on `kv260-01` and `kv260-02` (`board/disable-dp.sh`
installs a systemd unit that unbinds `fd4a0000.display`;
`board/jumbo-frames.sh` puts `mtu: 9000` in netplan; jumbo frames pass the
switch: 8972-byte pings between the two boards, 0 loss). **Not applied on
`kv260-03` and `kv260-04`**: those two boards ask for a sudo password over
ssh, which the session could not supply. The runs above were taken in that
mixed state (two boards at 9000 with the storm gone, two at 1500 with it),
so the jumbo-frame and interrupt-storm effects on the stream are not yet
separately attested; the code change (bytes, polling, pinning) is.

## What this does and does not establish

- The four-board stream is 27.0 t/s over 400 tokens, three runs within
  0.03 t/s of each other; at that depth a token is engine 19 + exchange
  13 + CPU 6 ms.
- The hidden state is bit-identical to `0010`'s (text byte-identical in
  all three runs); the engine contract is untouched.
- The exchange is still the second-largest term. What is left in it:
  ~0.24 ms of wire per all-reduce (12 KB each way), ~0.08 ms of fixed
  round-trip, and rank skew. Below that needs fewer bytes (fp16 partials,
  ~19 → 10 ms per token by the table above) or a faster link (2.5 GbE USB 3
  adapters and a 2.5 GbE switch would cut the wire term 2.5×; routing the
  boards through a Raspberry Pi would add a CPU hop and make it worse).

## Configuration

Board: Ubuntu 22.04, 5.15.0-1027-xilinx-zynqmp, macb driver (TSO not
available, interrupt coalescing not supported, `maxmtu 16365`), gcc 11.4,
`-O2`. Sockets: `TCP_NODELAY`, 1 MB send/receive buffers. `mesh_bench`
pinned with `PREFIX="taskset -c 2" bash board/tp/bench/run-mesh.sh <ips>`.

## Traps

- **`netplan apply` stopped NetworkManager on two boards and did not start
  it again** (journal: "caught SIGTERM, shutting down normally", then
  nothing). Both boards vanished from the network until NetworkManager was
  started over the serial console. Do not run `netplan apply` on these
  boards from a detached shell; edit the netplan file for the next boot
  and change the live interface with `ip link set eth0 down / mtu 9000 /
  up`, detached — the macb driver refuses an MTU change on a running
  interface (`Device or resource busy`), and NetworkManager's
  `nmcli con up` reports success without changing it.
- **The serial console numbering moved**: `/dev/ttyUSB5` and `/dev/ttyUSB1`
  are now `kv260-03` and `kv260-04`; boards 1 and 2 are on `ttyUSB13` and
  `ttyUSB9`. Read `hostname` before trusting a console.
- **Sudo differs between boards**: 1 and 2 are passwordless, 3 and 4 are
  not. Anything that needs root on 3 and 4 has to be run by hand.
- `mesh_bench`'s numbers move ±10% between runs; compare variants within
  one run, and trust the runtime's per-token split over 400 tokens.
