# CLAUDE.md

Guidance for Claude Code in this repository. Kept short on purpose: it loads
into every session. Detail lives in the docs, which are good — read them
instead of re-deriving.

Four Kria KV260 boards decode **one** LLM stream (Llama-3.2-1B-Instruct, all
tensors ggml Q4_0). Each has a weight-streaming int4 engine in its FPGA
(`hls/`, `hw/`), a driver (`board/kv260gemv/`), and a tensor-parallel runtime
with an OpenAI-compatible API on the LAN (`board/tp/`); a page on
al-nafisah.com (`/projects/llm-kv260`) describes it.

| where | what |
|---|---|
| `README.md` | what the project is |
| `docs/fabric-engine.md` | the design: engine, gemv2, DRAM banks, four boards |
| `docs/kv260-bringup.md` | board runbook, board/MAC table, **the traps** |
| `docs/buildlog/` | every measured number + raw log under `evidence/` |
| `docs/measurement.md` | attested vs cited |

## Decisions (do not relitigate)

- The engine consumes **ggml Q4_0 weights and Q8_0 activations**: the GGUF's
  own block bytes, unmodified, re-laid-out per port when the runtime loads
  them. Integer in-block accumulation makes bit-exactness
  free. The contract is `src/llm_kv260/q4_0.py`; one differing byte between
  fabric and contract is a failure, no tolerance.
- Summation order is the contract's, not llama.cpp's: int32 block dot, two
  float32 multiplies, eight sequential lanes (block b into lane b mod 8), then
  a binary tree. The engine will never bit-match a CPU run and is not meant
  to; the CPU is only a float-precision sanity check.
- Four boards, **one stream** (tensor parallelism), not four replicas. The
  exchange is a TCP full mesh; a UDP transport lost packets and was removed.
- The API serves **one prompt at a time in arrival order**. Latency and rate
  of that one stream are the goal, not concurrency.
- Board software: no PYNQ. u-dma-buf for memory, UIO for registers, `xmutil`
  for the overlay. The overlay must carry the `xlnx,afi-fpga` node — without
  it the PS AXI master stays 128-bit, a 32-bit control port loses most
  register writes, and the engine "finishes" in 2 µs.
- Vitis HLS first; hand RTL only where HLS cannot be made fast enough.

## Hard constraints (violate these and it is silently wrong)

- `k % 256 == 0`, `k <= 8192`, `rows % 32 == 0` (the contract admits `% 16`,
  so gemv2 skips the `_n16` vector cases).
- Activation buffer is `int8[K]` then `float32[K/32]` (`pack_x_for_engine`).
- The four port streams of a matrix must **not all start at the same offset
  modulo 64 KB**, or they collide in the DRAM banks (12.7 vs 10.3 GB/s on the
  old engine, buildlog 0012). The runtime 64 KB-aligns each quarter, shifts
  ports 1 and 3 by 16 KB (two offsets, 0 and 16 KB, alternate) and passes
  four addresses (`gemv_dev_start4`). Pads of 256 B to 8 KB were not enough.
- gemv2 weight layout: per port, row i pairs with row i + rows/8 and their
  144-byte chunks alternate (`gemv2_chunk_offset`); every call names the next
  matrix; a fresh process drains the FIFOs first.
- `hls/int4_core/gemv_core.h` is the shared MAC — **changing it changes all
  eight MAC units**.
- `xgemv_*_hw.h` next to the driver are HLS-generated copies: re-copy them
  whenever a kernel's argument list changes.
- The cache invalidate *before* an engine call stays: `rope()` writes q in
  place in the result buffer, and without it the text is garbage.

## Shape of the code

- Bit-exactness chain, in build order, each stage checked against the one
  before it: `src/llm_kv260/q4_0.py` (contract + golden hash in `tests/`) →
  `gen_vectors.py` → HLS kernel + C testbench → `build.tcl`/`package.sh` →
  on-board `bench_gemv2` (`EXACTNESS: PASS`) → runtime.
- `board/tp/tp_llama.cpp` is one file (sections: memory, model, engine, comm,
  ops, stats, decode, server, main). It links libllama only for the tokenizer,
  GGUF reader and chat template; everything between matmuls is NEON intrinsics
  on one A53 core — the board's GCC 11 at `-O2` vectorizes nothing by itself,
  and fixed-count loops need `#pragma GCC unroll` or accumulators spill. KV
  cache is `[layer][kv head][pos][head_dim]`, one head contiguous; the
  interleaved layout was 1.3–1.8× slower in attention (`0010` microbench).
- Tensor parallelism: column-parallel matrices (q, k, v, gate, up, lm_head)
  shard by output rows; row-parallel ones (attn_output, ffn_down) by input
  columns — so one all-reduce after attention and one after the MLP per layer,
  plus an argmax exchange per token. An all-reduce is reduce-scatter +
  all-gather with polled reads; partials sum in rank order on every rank so
  the hidden state stays bit-identical everywhere. Rank r owns query heads
  8r..8r+7, so attention needs no traffic. Rank 0 serves HTTP; >8 queued
  prompts gets 429. The runtime pins to CPU 2 (`--cpu`): CPU 0 has a
  DisplayPort interrupt storm, CPU 3 the NIC's IRQ.
- Engine selection is automatic: the runtime opens whichever UIO device the
  loaded overlay provides (`gemv_mac`, or the older `gemv_q4_0`) and lays the
  weights out for it. Only the gemv2 overlay is built from this repo.
- `bench_gemv2` is the only bench: exactness, then cold and prefetched GB/s.
  It needs the gemv2 overlay and the `_n32` vector cases.

## The API stays on the LAN

Since 2026-11-04 the system is **LAN-only**: the API answers on board 1:8080
(`http://10.42.0.137:8080` while the lease holds) and nothing routes to it
from the internet. The Cloudflare tunnel (`cloudflared-kv260`) and the
logging proxy are stopped and disabled; do not re-enable them unless the
user asks. The project page (https://al-nafisah.com/projects/llm-kv260, in
the site repo `al-nafisah/AbdullahAlNafisah.github.io`) describes the work
and has no live chat. Local clients: `board/tp/chat.py` (terminal) and
`local/` (a page on 127.0.0.1:8090).

- `local/index.html` pins marked, DOMPurify and KaTeX to a version **and** a
  SHA-384 `integrity` hash. Raising a version without recomputing the hash
  leaves the chat silently plain. Hash: `curl -sf URL | openssl dgst -sha384 -binary | openssl base64 -A`.
- `/stats` feeds the board cards. "Memory BW (token avg)" is shard bytes over
  the **whole** per-token time — dividing by the engine term alone read
  ~18.5 GB/s, above what the DRAM can do, because gemv2 prefetches during the
  exchange. `power_w` is the K26 module's INA260 only, sampled once per
  generated token, so an idle `/stats` repeats a stale value; the sampled
  numbers are in buildlog 0013.
- The API is open by design (no auth, CORS `*`), which is why it stays on
  the LAN. Three holes are known and unfixed, each needing a rebuild +
  restart: cpp-httplib's 100 MB default body, a stream whose client left runs
  to the end holding the slot (nothing checks `sink.is_writable()`), and one
  caller can hold all 8 queue slots.

## Machines

- **PC**: 15 GB RAM, 6 cores, Vivado/Vitis HLS 2022.2 at `/opt/Xilinx`. Run
  Vivado **detached** (`setsid nohup … &`) with **three** jobs — six parallel
  synthesis jobs exhaust memory and kill background tasks. Vitis HLS
  `csim_design` cannot link on this glibc; build the testbench with system
  `g++` against the Vitis headers (see Commands). User's shell is fish, the
  tool shell is zsh — **run `board/tp/*.sh` with `bash`** (zsh's 1-indexed
  arrays and unsplit variables have broken the launchers twice).
- **Boards**: `kv260-01..04`, user `ubuntu`, 4× A53, 3.8 GiB RAM, no swap.
  Addresses are DHCP and **move** — `docs/kv260-bringup.md` has the MAC per
  board, and `ip neigh | grep 00:0a:35` is always right. Boards 1 and 2 have
  passwordless sudo; **3 and 4 need the user's password**, and sudo rights
  follow the hostname, not the address. Consoles via `board/serial-cmd.py`
  (numbering follows plug order — read `hostname` first); they are not always
  cabled, so fix networking before unplugging them. MTU 9000 and the unbound
  DisplayPort survive a reboot; **overlay, memory pools and the AXI width do
  not**: `sudo bash ~/board/load-overlay.sh ~/gemv2`, which every board now
  allows without a password, so `bash board/tp/recover.sh` recovers all four. CMA is fragmented — one
  allocation above ~512 MiB fails, so pools are 512 + 256 + 128 MiB.
- **Board builds happen on the board**: `~/kv260gemv` (driver + bench),
  `~/board/tp` (runtime, needs `~/llama.cpp` via `board/board-llama-build.sh`,
  ~20 min detached). The driver is on each board **twice** — the runtime
  compiles `~/board/kv260gemv` (`GEMV` in the Makefile), the benches build in
  `~/kv260gemv` — so a driver change goes to both.

## Conventions

- **Attested vs cited numbers never blend** (`docs/measurement.md`): every
  number is measured here with its raw log under `docs/buildlog/evidence/`,
  or attributed. A failed build is a result.
- Build logs are `NNNN-title.md` with sections Status / Result / What this
  does and does not establish / Configuration / Traps; evidence is
  `evidence/NNNN-*.txt` (`.txt` because `*.log` is gitignored).
- Weights are never committed (`models/` gitignored). Quantize with
  `llama-quantize --output-tensor-type q4_0 --token-embedding-type q4_0 …
  Q4_0`; without those flags `output.weight` stays Q6_K and the engine cannot
  take it.
- Build outputs are never committed either — per-directory `.gitignore`s use
  wildcards (`out*/`, `ip_repo*/`, `*_prj*/`) so `GEMV_VARIANT` experiments
  are covered.
- A per-token CPU number depends on context depth: say what position it was
  taken at, and **measure with 400 tokens, not 64**.
- **A busy board gives a wrong number, not a slow one** (unattended-upgrades
  turned 305 t/s into 21). Check idle before believing a number.
- `bench_gemv2` defaults to `udmabuf0`, which holds a resident runtime's
  shard: stop the runtime first or pass `udmabuf2`.
- `ssh` in the background needs `-n`; a detached remote process needs
  `setsid nohup … < /dev/null &`, or the session hangs.
- `PL0_REF_CTRL` rounds 300 MHz to 333; neither helps, the DDR saturates at
  ~12.7 GB/s (buildlog 0012).
- **Never `netplan apply` on a board from here** — it stopped NetworkManager
  and the board was gone until the serial console started it. Edit the file
  for next boot; change the live MTU with `ip link set eth0 down/mtu/up`,
  detached (macb refuses it while up). A netplan file naming eth0 must also
  say `dhcp4: true`, or the board boots with no address.

## Commands

```bash
# PC: environment and the contract
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest tests/ -q                                   # the contract
.venv/bin/python hls/int4_core/gen_vectors.py hls/int4_core/vectors     # golden vectors

# PC: engine -> IP -> bitstream -> firmware directory
g++ -std=c++14 -O1 -I/opt/Xilinx/Vitis_HLS/2022.2/include -Ihls/int4_core -Ihls/gemv2 \
  -o /tmp/tb2 hls/gemv2/gemv_fetch.cpp hls/gemv2/gemv_mac.cpp hls/gemv2/tb_gemv2.cpp && /tmp/tb2 hls/int4_core/vectors
(cd hls/gemv2 && source /opt/Xilinx/Vitis_HLS/2022.2/settings64.sh && vitis_hls -f run_hls.tcl)   # GEMV_CLK_NS
(cd hw/kv260_gemv2 && source /opt/Xilinx/Vivado/2022.2/settings64.sh && vivado -mode batch -source build.tcl)
hw/kv260_gemv2/package.sh                                              # -> out/gemv2/
# GEMV_VARIANT=tag / GEMV_MHZ=… keep a Vivado experiment in *_tag dirs

# board: load the engine, prove it, build the runtime
sudo bash ~/board/load-overlay.sh ~/gemv2                              # app name is the .bit.bin's
(cd ~/kv260gemv && ./bench_gemv2 ~/vectors udmabuf2)                   # exactness + cold/prefetched GB/s
(cd ~/board/tp && make)

# PC: run and serve (bash, not zsh) — confirm addresses first, they move
bash board/tp/run-tp.sh "prompt" 400 10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114
bash board/tp/serve-tp.sh 10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114   # API on board 1:8080
bash board/tp/serve-tp.sh stop 10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114
bash board/tp/recover.sh                                               # after a power cut: boards, overlays, ranks, LAN check
.venv/bin/python board/tp/api-bench.py http://10.42.0.137:8080 --n 400 --repeats 3
PREFIX="taskset -c 2" bash board/tp/bench/run-mesh.sh 10.42.0.137 10.42.0.218 10.42.0.217 10.42.0.114
python3 board/tp/chat.py                                               # chat from the terminal
bash local/serve.sh                                                    # local page -> 127.0.0.1:8090
python3 board/tp/bench/session.py OUT <four boards>                    # recorded session: rate + module power (0013)
```

`run-tp.sh` honours `MODEL` (its default is the **base** model; the served
one is `MODEL=~/llama-3.2-1b-instruct-q4_0-pure.gguf`), `OUT`, `LIMIT`,
`STAGGER` and `BIN=./tp_llama.old` (A/B against a kept binary); it passes
`TP_DEBUG` through, but the current binary ignores it. `serve-tp.sh` honours
`MODEL` (default: the Instruct model) and `HTTP_PORT`. Ranks log to `~/tp-serve.log` on each board.
