# Edge FPGA-based LLM

Four AMD Kria KV260 boards run one language model together and answer over
an OpenAI-compatible API. Try it: **https://al-nafisah.github.io/llm-kv260-demo/**

## What is implemented, in plain words

A language model produces text one token at a time, and for every token it
has to read all of its weights. For the model used here (Llama-3.2-1B-Instruct,
4-bit, 663 MiB) that read is the whole cost of a token; the arithmetic is
small. So the work is: read weights fast, and split the reading across boards.

1. **A weight-reading engine in each board's FPGA** (`hls/`, `hw/`). It pulls
   4-bit weights out of memory through four ports at ~13 GB/s and multiplies
   them by the current activation on the way past. It computes exactly what
   `src/llm_kv260/q4_0.py` says, bit for bit, and that is checked on every
   board (`board/kv260gemv/bench_gemv2`).
2. **A small runtime on each board's ARM cores** (`board/tp/tp_llama.cpp`)
   that does everything around the engine: attention, normalisation, the
   tokenizer, sampling. Four boards each hold a quarter of every weight
   matrix, run their quarter on their own engine, and exchange partial
   results over a gigabit switch twice per layer. All four print the same
   text; one board alone does ~10 tokens/s, four together ~36, and the rate
   holds over a long reply (400 tokens: 36.3 t/s).
3. **An OpenAI-compatible API** (`tp_llama --serve`) on board 1, one prompt at
   a time in arrival order so the one stream gets the whole machine, with
   `/stats` reporting what each board is doing: engine time, exchange time,
   CPU, RAM, temperature and power (about 5 W per board while decoding).
4. **A public page** (`web/`, GitHub Pages) that chats with it and shows the
   four boards live.

Measured numbers with their raw logs are in `docs/buildlog/`; the design is
in `docs/fabric-engine.md`; setting up a board is `docs/kv260-bringup.md`.

## Layout

| path | what |
|---|---|
| `src/llm_kv260/q4_0.py`, `tests/` | the bit-exact definition of the engine's result, and its tests |
| `hls/int4_core/` | the Q4_0 block layout, the shared MAC, and the golden vectors |
| `hls/gemv2/` | the engine (Vitis HLS C++): fetch, UltraRAM FIFOs, eight MAC units |
| `hw/kv260_gemv2/` | Vivado block design, bitstream packaging, device-tree overlay |
| `board/kv260gemv/` | driver: contiguous memory (u-dma-buf), registers (UIO), benchmark |
| `board/tp/` | the tensor-parallel runtime, the API server, launch and publish scripts |
| `board/*.sh`, `board/*.py` | bringing a fresh board up and loading the engine |
| `web/` | the test page |
| `docs/` | design, runbook, build logs with evidence |

## Rebuilding from source

```bash
# PC: engine -> IP -> bitstream -> firmware directory
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest tests/ -q
.venv/bin/python hls/int4_core/gen_vectors.py hls/int4_core/vectors
(cd hls/gemv2 && source /opt/Xilinx/Vitis_HLS/2022.2/settings64.sh && vitis_hls -f run_hls.tcl)
(cd hw/kv260_gemv2 && source /opt/Xilinx/Vivado/2022.2/settings64.sh && vivado -mode batch -source build.tcl)
hw/kv260_gemv2/package.sh

# each board: engine, driver, runtime (see docs/kv260-bringup.md)
sudo bash ~/board/load-overlay.sh ~/gemv2
(cd ~/kv260gemv && make && ./bench_gemv2 ~/vectors)
(cd ~/board/tp && make)

# PC: one stream across the four boards, then the API
board/tp/run-tp.sh "The three primary colors are" 64 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
board/tp/serve-tp.sh 10.42.0.137 10.42.0.217 10.42.0.218 10.42.0.114
```

Weights are never committed. The model is the GGUF of Llama-3.2-1B-Instruct
quantized with `llama-quantize --output-tensor-type q4_0 --token-embedding-type q4_0 ... Q4_0`.
