# 0006 — The fabric engine: bit-exact on hardware, 11.6 GB/s per board

**Status: Attested.** 2026-09-13, `kv260-01` (later all four boards, `0007`),
Kria Ubuntu 22.04 (5.15.0-1027-xilinx-zynqmp), Vitis HLS + Vivado 2022.2.
Raw logs: `evidence/0006-*`. Design: [../fabric-engine.md](../fabric-engine.md).
The `gemv_q4_0` kernel and `bench_gemv` measured here left the tree when
gemv2 replaced them (`0012`); the datapath and the vectors are now in
`hls/int4_core/`.

## Result

`bench_gemv` (`board/kv260gemv/`), `evidence/0006-bench-gemv-kv260-01.txt`:

- Bit-exact against the contract's vectors (`hls/int4_core/gen_vectors.py`)
  on hardware: K = 256, 2048, 8192, every row, every byte.
- Weight streaming at 250 MHz through four HP ports, best of N:

| shape (Llama-3.2-1B) | bytes | time | GB/s |
|---|---|---|---|
| q_proj 2048×2048 | 2.36 MB | 0.232 ms | 10.16 |
| k_proj 512×2048 | 0.59 MB | 0.051 ms | 11.67 |
| gate 8192×2048 | 9.44 MB | 0.877 ms | 10.76 |
| down 2048×8192 | 9.44 MB | 0.813 ms | 11.60 |
| lm_head 128256×2048 | 147.75 MB | 12.761 ms | 11.58 |

That is 60% of the DDR4 peak (19.2 GB/s). llama-fpga (Cited,
arxiv 2502.10659) reached 84.5% of peak on this board, bare-metal.

## What it took (in the order it went wrong)

1. **The contract first.** `src/llm_kv260/q4_0.py`: ggml Q4_0 × Q8_0, int32
   in-block dot, `(float(sumi)*d_w)*d_x`, eight lanes, tree; codec checked
   byte-for-byte against gguf-py.
2. **HLS, three cuts.** First synthesis: 90% LUTs and II=7. The unpacker's
   1152-bit variable-position buffer was 15K LUTs per engine (now a two-beat
   window: 1.7K); the lane shift register could not hide the float adder
   (now a two-port lane RAM with `DEPENDENCE distance=8`: II=1). Final:
   II=1 everywhere, 3.63 ns estimated at 4 ns, 50% LUT, 13% DSP
   (`evidence/0006-hls-csynth.txt`).
3. **The C testbench cannot link inside Vitis HLS on this PC** (its binutils
   2.37 rejects modern glibc's `.relr.dyn`). The same testbench built with
   the system `g++` against the Vitis headers passes; `hls_half.h` was
   replaced by an explicit float16→float32 converter so nothing but libc is
   needed.
4. **Vivado, three builds.** Connection automation, asked to attach five
   masters to HPC0, chained four extra SmartConnects whose checkpoints never
   linked (DRC INBB-3 black boxes); replaced by one explicit 5:1
   SmartConnect. Then the AXI read FIFOs — 32 outstanding × 256 beats ×
   128 bits = 128 KB per port — over-filled the block RAM (316 of 288);
   16 × 64 beats is still 8× the bandwidth-delay product and fits.
   Final: 32,228 LUTs (27.5%), 48 of 144 BRAM tiles, 228 DSPs,
   WNS −0.049 ns on 122 endpoints (the 32-bit `rows × blocks` multiply
   in `mac`), accepted because the exactness test passes on hardware
   (`evidence/0006-vivado-timing-summary.txt`).
5. **The board's memory.** `cma=1000M`, but one allocation over ~512 MiB
   fails with 989 MiB free (boot-time fragmentation); three u-dma-buf pools
   (512 + 256 + 128 MiB) work.
6. **The engine finished in 2 µs and wrote nothing.** Register readback
   showed `k = 0`: the Kria boot firmware leaves M_AXI_HPM0_FPD at 128 bits
   (`FPD_SLCR.AFI_FS = 0xa00`) and the design's control port is 32 bits, so
   only writes at 16-byte-aligned offsets landed. Fixed with an
   `xlnx,afi-fpga` node in the overlay.
   `evidence/0006-bench-gemv-kv260-01-before-afi-fix.txt` is the failure.

A llama.cpp backend was written first to prove the engine end to end (9.03
tokens/s on one board against 2.98 on its CPU, with 40% of every token lost
to the scheduler starting the CPU backend 86 times). It is not part of the
running system; the runtime that replaced it is `0007`.

## Quality of the model being served

Full wikitext-2 test, ctx 512, PC (`evidence/0006-ppl-llama-3.2-1b-*.txt`):
Llama-3.2-1B F16 **10.19**, all-Q4_0 **11.64**. Forcing the output tensor to
Q4_0 (rather than llama-quantize's default Q6_K) costs 1.1% of perplexity and
keeps the largest matmul of every token on the engine.
