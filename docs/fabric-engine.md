# The fabric engine (phase 4)

What runs in the PL, what runs on the PS, and why the line is where it is.
Numbers here are **Attested** unless marked Estimated or Cited; the raw logs
are in [buildlog/evidence](buildlog/evidence), under the number of the buildlog
each statement names.

## The shape of the problem

Decoding one token of Llama-3.2-1B reads every weight once: 695.1 MB
(662.9 MiB) at Q4_0 (`buildlog/0013`). Nothing else in the token is remotely
that large. The KV260's DDR4 interface peaks at 19.2 GB/s, and
[llama-fpga](https://arxiv.org/html/2502.10659v1) reports 84.5% bandwidth
utilization on this board, bare-metal: its measured decode rate over the rate
19.2 GB/s would allow, 4.9 against 5.8 tokens/s for Llama2-7B at 4 bits (all
Cited, from that paper). A token's time is therefore set by how fast the
fabric can stream those 695.1 MB, and the whole design question is how fast
one board's engine streams, how little the PS adds on top, and how the
stream splits across boards.

## What is in the PL

`hls/int4_core/` holds the Q4_0 block layout and the MAC; `hls/gemv2/` the
kernels that run it. The first engine — one Vitis HLS kernel, `gemv_q4_0`,
four identical engines — is what `buildlog/0006` measured; gemv2 replaced it
and its sources left the tree. The stages are the same:

| stage | per engine | what it does |
|---|---|---|
| `unpack` | 1 beat/cycle | 128-bit AXI beats in, 144-bit Q4_0 blocks out. 8 blocks are exactly 9 beats, and block *j* always sits in beats *j*, *j*+1 at byte offset 2*j*, so it is a two-beat window and an 8-way mux |
| `mac` | 1 block/cycle | 32 int4×int8 products summed in int32; `(float(sumi) * d_w) * d_x`; eight float32 lanes in a small two-port RAM; the tree over the previous row's lanes read out one per cycle |
| output | 4 rows/beat | float32 results packed four to a 128-bit write |

Each engine owns one PS high-performance port (`S_AXI_HP0..3_FPD`, 128-bit)
and streams a contiguous quarter of the matrix's rows. The bytes are the
model's own Q4_0 blocks, unconverted, but not read in place: the runtime
copies them out of the GGUF into engine memory at load (for gemv2 in a
different order, below). The activation vector (int8[K] + float32[K/32]) is
loaded once per call into four private BRAM copies. Activations in and
results out share `S_AXI_HPC0_FPD` through a SmartConnect; the traffic there
is kilobytes per call.

Synthesis of the first engine (`gemv_q4_0`, Vitis HLS 2022.2 estimate, `xck26-sfvc784-2LV-c`, 4 ns with 0.4 ns uncertainty, `buildlog/0006`; the routed gemv2 build is in the README):

| | |
|---|---|
| every loop | II = 1 |
| estimated clock | 3.625 ns |
| LUT / FF / DSP / BRAM | 59,140 (50%) / 36,994 (15%) / 164 (13%) / 72 (25%) |

Two rewrites got it there. The first unpacker held a 9-beat, 1152-bit buffer
with a variable-position write and cost **15,189 LUTs per engine**; the
two-beat window costs 1,681. The first accumulator
used a shift register of eight lanes, which HLS could only schedule at
**II = 7** because the float adder's result is needed one iteration later
by the shift; the lane RAM with `DEPENDENCE ... distance=8` states the real
dependence distance and gets II = 1.

### Why the engine consumes ggml's Q4_0 with int8 activations

A first contract pinned int4 weights × **float32** activations. A fabric
that multiplies float32 at DRAM speed needs ~32 float multiplies per cycle
per engine, ~600 DSPs (Estimated) and a float adder tree whose order also
has to be pinned. With int8 activations the 32 in-block products sum in
int32, where order is irrelevant and bit-exactness is free; only one float
multiply and one float add per *block* remain. That is a ~10× cheaper
datapath (Estimated), and it is what let all four engines fit with half the
fabric to spare.

Q4_0 (block 32, float16 scale) is what a GGUF file already holds, so the
engine reads the model's own block bytes and no value is converted; the
runtime only copies and reorders the blocks at load.
`src/llm_kv260/q4_0.py` is the contract.

## What is on the PS

`board/kv260gemv/` — the driver, ~200 lines of C, no framework:

- **u-dma-buf** gives physically contiguous, CPU-cached memory from CMA and
  its physical address. Coherence is explicit and cheap: `dc cvac` before the
  engine reads what the CPU wrote, `dc civac` before the CPU reads what the
  engine wrote, both legal from user space on arm64. No syscall per op.
- **UIO** maps the engine's AXI-Lite registers (offsets from the headers Vitis
  HLS generates, copied next to the driver; nothing checks the copies, so
  they are re-copied whenever a kernel's argument list changes).
- `gemv_dev_run(w, x, y, k, rows)` splits rows four ways, writes nine
  pointers and two sizes, sets `ap_start`, polls `ap_done`.

`board/tp/tp_llama.cpp` — the runtime: everything that is not a matmul
(attention, RMSNorm, RoPE, SiLU, sampling, the tokenizer through libllama)
as plain loops on one A53 core, with no scheduler between engine calls;
q|k|v and gate|up are each one engine call, so a token is 65 engine passes.
It is also the tensor-parallel runtime and the API server (next sections).

Memory: `cma=1000M` is on the Kria Ubuntu command line, but one allocation
above ~512 MiB fails even with 989 MiB free (the area is fragmented by boot
allocations). Three pools — 512 + 256 + 128 MiB — succeed.

## Where the bytes go in DRAM

The driver used to place engine m's quarter of a matrix at
`w + m × quarter_bytes`. For the gate|up and down shards that stride is a
multiple of 64 KB (for the q|k|v, o and lm_head shards it is not), so their
four streams sat at the same offset modulo 64 KB and collided in the DRAM
banks: 10.3 GB/s. The runtime now allocates each quarter 64 KB-aligned and
shifts ports 1 and 3 by 16 KB, so ports 0 and 2 start at 0 and ports 1 and 3
at 16 KB modulo 64 KB, and passes the four addresses through
`gemv_dev_start4` (`gemv2_fetch_start` on gemv2); the same shapes then
stream at 12.7 GB/s (`buildlog/0012`). Of the 16 KB-step placements the
bench tried, only those with all four quarters at the same offset collide;
every other one, with two distinct offsets or four, reads 12.6–12.8 GB/s
on gate|up and down, while spacing consecutive quarters by 256 B to 8 KB,
which also gives four distinct offsets, reads 10.3–11.9. The clock is not
the lever: the same engine at 333 MHz reads no more than 12.0 GB/s on any
shape or placement tested. Under Linux the four PL read streams reach
12.7 GB/s on the old engine and 13.0 GB/s for a token's 65 gemv2 calls back
to back (12.96–13.06 on the four boards, `buildlog/0013`); the DDR4
interface peak is 19.2 GB/s (Cited, above). The measured rate is the number
a token's weight stream divides.

## gemv2: the next matrix loads while the CPU works

`hls/gemv2/` is the same datapath cut in two, with UltraRAM between:

| kernel | what it does |
|---|---|
| `gemv_fetch` | four DRAM readers, one per HP port; port p's stream is dealt chunk by chunk (144 B) to FIFOs pa and pb |
| eight AXI-Stream FIFOs | 16384 × 128 bit each in UltraRAM (8 URAM each, all 64 of the device): 2 MB of the next matrix |
| `gemv_mac` | eight MAC units (`gemv_core.h`, the same arithmetic), each draining one FIFO and writing its rows of y |

The runtime names, with every call, the matrix that comes next; its fetch
starts the moment this call's MAC is done and runs while the CPU does
attention or the exchange, so up to 2 MiB of every matrix (the first 512 KiB
of each port's quarter) is on chip when its activations arrive, as much as
the time since the previous call allowed. Two MAC units per port because a
full FIFO only helps if it drains faster than DRAM refills it: one unit per
port consumes 16 GB/s against a 12.7 GB/s DRAM, two consume 32. The layout that
makes this a single stream per port pairs row i with row i + rows/8 and
alternates their chunks (`gemv2_chunk_offset`); the runtime copies the GGUF
chunk by chunk at load. `rows_per_mac = 0` is a drain command: a fresh
process empties whatever the previous one left in the FIFOs. The runtime
picks the engine by the UIO device the overlay provides (`gemv_mac` or
`gemv_q4_0`) and lays the weights out accordingly, so one binary runs on
either overlay. Measured results: `buildlog/0012`.

## Building and loading

```bash
# PC: kernels -> IP -> bitstream -> firmware directory
GEMV_HLS_MODE=export vitis_hls -f hls/gemv2/run_hls.tcl            # both kernels
(cd hw/kv260_gemv2 && vivado -mode batch -source build.tcl)        # ~35 min
hw/kv260_gemv2/package.sh                                           # bootgen (+ dtc if present)
scp -r hw/kv260_gemv2/out/gemv2 board/ ubuntu@kv260-01:~/
# no dtc on this PC: compile the overlay on the board
ssh ubuntu@kv260-01 'cd gemv2 && dtc -@ -I dts -O dtb -o gemv2.dtbo pl.dtsi'

# board
sudo bash ~/board/load-overlay.sh ~/gemv2
(cd ~/kv260gemv && make && ./bench_gemv2 ~/vectors)                # exactness + GB/s
bash ~/board-llama-build.sh                                        # libllama for the runtime, ~20 min
(cd ~/board/tp && make)
```

The C testbench is compiled with the system `g++` against the Vitis HLS
headers (`g++ -std=c++14 -I/opt/Xilinx/Vitis_HLS/2022.2/include ...`)
because `csim_design` cannot link on this PC's glibc; the two are the same
code under the same compiler semantics, and the vectors come from the
contract (`hls/int4_core/gen_vectors.py`).

## Four boards, one stream

`board/tp/tp_llama.cpp` is a second runtime on the same engine: no llama.cpp
scheduler, and tensor-parallel across boards. The single-board llama.cpp
path spends 40% of a token starting the CPU backend between matmuls; this
runtime is one loop on one core, in NEON intrinsics (`buildlog/0010`: the
board's GCC 11 vectorizes none of it at `-O2`, and written scalar the
attention alone cost 0.12 ms per context position per token), and it
shards every matrix across ranks:

| matrix | sharded by | on rank r | traffic |
|---|---|---|---|
| q, k, v, gate, up, lm_head | output rows | rows `[rN/n, (r+1)N/n)` — a contiguous byte range of the Q4_0 file | none |
| attn_output, ffn_down | input columns | columns `[rK/n, (r+1)K/n)` — 18-byte blocks sliced from each row | **all-reduce** of the partial product |
| attention | heads | query heads `8r..8r+7` and the KV heads they read | none |

Two all-reduces per layer and one argmax exchange per token (each rank's
top 40 candidates by default, as 8-byte value and index pairs, plus 36 bytes
of board stats: 356 bytes per rank), over a full mesh of TCP connections
(`TCP_NODELAY`; a UDP version with its own retransmits lost too much under
the 18-packet bursts of a 4-way exchange).
Each all-reduce is a reduce-scatter (rank r receives the others' partials
of its quarter of the vector and sums them in rank order) and an
all-gather of the summed quarters: half the bytes of a plain all-gather,
which mattered because serializing the bytes at 1 Gbit/s was about half the
time of the 8 KB all-gather it replaced (`buildlog/0011`). Reads are
non-blocking and polled, and the process pins itself to CPU 2, away from
the DisplayPort interrupt storm on CPU 0 and the NIC's IRQ on CPU 3. The
sums are the same float32 additions in the same order on every rank, so the
replicated state is bit-identical everywhere. q|k|v and gate|up are each
one engine call, so a token is 65 engine passes instead of 113. The KV cache
is laid out `[layer][kv head][pos][head_dim]` so attention over one KV head
reads its K rows once, in position order, and its V rows in four sequential
passes of 16 columns; the 4 query heads of a KV group share each pass.

```bash
board/tp/run-tp.sh "The three primary colors are" 64 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
```

Measured (`buildlog/0007-four-boards-tensor-parallel.md`): 1 board 10.49
t/s, 2 boards 17.4, **4 boards 21.4–23.8**, identical text in the logged
prefix, on 64-token runs. The CPU term grows with the context: averaged over
a 400-token generation it was 32 ms of a 69 ms token before the NEON ops and
6–7 ms of a 43–44 ms token after (`buildlog/0010`): engine 19.0–22.1 ms,
exchange 16.5–20.7 ms, CPU 5.9–7.0 ms, 22.7–23.1 t/s over 400 generated
tokens. With the reduce-scatter exchange (`buildlog/0011`):
engine 19, exchange 13, CPU 6 ms, 27.0 t/s over 400 tokens. With gemv2 on
every board and jumbo frames everywhere (`buildlog/0012`): engine 10,
exchange 12, CPU 6 ms, **36.2–36.4 t/s** in three runs; three more runs of
the same build gave 34.7–36.7 t/s (`buildlog/0013`).

## Measured

- Engine (`buildlog/0006`, `0007`): bit-exact on all four boards; 10.2–11.7
  GB/s on kv260-01, 10.2–12.1 across the four.
- Bank placement and NEON quantization (`buildlog/0012`): quarters that do
  not all start at the same offset modulo 64 KB stream 12.7 GB/s on gate|up
  and down; the four-board stream runs at 29.2 t/s over 400 tokens, engine
  15.8–16.2 ms per token, text byte-identical to `0011`. 333 MHz: at most
  12.0 GB/s, not a lever.
- gemv2 on all four boards, with MTU 9000 and the DisplayPort storm gone on
  all four (`buildlog/0012`): **36.2–36.4 t/s** over 400 tokens in three
  runs, per token engine 9.9–10.0 ms, exchange 11.7–12.0, CPU 5.9–6.1, text
  byte-identical to `0011`. Bit-exact on the contract's vectors on every
  board, including one rank's shard shapes. Three more runs of the same
  build (`buildlog/0013`): 34.7–36.7 t/s.
- Four boards, one stream (`buildlog/0007`): 1 board 10.49 t/s, 2 boards
  17.4, 4 boards 21.4–23.8, identical text in the logged prefix, on 64-token
  runs. On the current build (`buildlog/0013`, 400 tokens at positions
  24–423): 1 board 13.39–13.53 t/s, 2 boards 21.65–21.87, 4 boards
  34.71–36.69.
- The CPU term (`buildlog/0010`): scalar ops cost 0.12 ms per context
  position per token; in NEON, with a per-head KV cache layout, a 400-token
  generation runs at 22.7–23.1 t/s with 6–7 ms of CPU per token on average
  (was 14.5 t/s and 32 ms).
- The exchange (`buildlog/0011`): a four-way 8 KB all-gather is 0.38 ms
  pinned, of which ~0.2 ms is wire; as reduce-scatter + all-gather with
  polled reads the all-reduce took 0.34 ms in the bench's own variant and
  0.36 ms in the committed code in the network state of the 0011 runs
  (0.28 ms once all four boards ran MTU 9000 without the DisplayPort storm,
  `buildlog/0012`); the stream runs at 27.0 t/s over 400 tokens (three runs
  within 0.03 t/s), exchange 13 ms per token.
- Through the API (`buildlog/0009` through the public URL, `0010`, `0011`,
  `0012`): 20.8–21.9 t/s to a client before the NEON ops, 24.7–28.0 after,
  29.4 after the exchange change, 39.95–40.09 with gemv2, on a 67-token
  reply.
