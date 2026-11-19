# Building it from source

Running the API and chatting with it from a PC on the same network:

```bash
bash board/tp/serve-tp.sh <board 1> <board 2> <board 3> <board 4>   # API on board 1:8080
python3 board/tp/chat.py                       # terminal chat (/stats, /reset, /quit)
bash local/serve.sh                            # local page on http://127.0.0.1:8090
```

The API has no authentication, so keep it on the local network.

## Toolchain and steps

Toolchain: Vitis HLS and Vivado 2022.2 on the PC, Python 3.11 with `uv`.
The boards run Kria Ubuntu 22.04 (kernel 5.15.0-1027-xilinx-zynqmp) with
gcc 11.4. The runtime links llama.cpp's library on each board for the
tokenizer, the GGUF reader and the chat template; that copy came from a
source tarball, and its revision was not recorded. The shipped bitstream has
md5 `81e7b63d430956c99de998ab90bfffda`.

```bash
# PC: the contract, the engine, the bitstream
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest tests/ -q
.venv/bin/python hls/int4_core/gen_vectors.py hls/int4_core/vectors
(cd hls/gemv2 && source /opt/Xilinx/Vitis_HLS/2022.2/settings64.sh && vitis_hls -f run_hls.tcl)
(cd hw/kv260_gemv2 && source /opt/Xilinx/Vivado/2022.2/settings64.sh && vivado -mode batch -source build.tcl)
hw/kv260_gemv2/package.sh

# each board (kv260-bringup.md): engine, proof, runtime
sudo bash ~/board/load-overlay.sh ~/gemv2
(cd ~/kv260gemv && make && ./bench_gemv2 ~/vectors)        # EXACTNESS: PASS
(cd ~/board/tp && make)

# PC: one stream across the four boards, then the API (bash, not zsh)
bash board/tp/run-tp.sh "Write a detailed history of the Roman Empire, from its founding to its fall, covering the key events and figures." 400 <board 1> <board 2> <board 3> <board 4>
bash board/tp/serve-tp.sh <board 1> <board 2> <board 3> <board 4>

# PC: the measurement session of buildlog 0013 (rates, module power, CPU baseline)
python3 board/tp/bench/session.py out/ <board 1> <board 2> <board 3> <board 4>
python3 board/tp/bench/session-report.py out/
```

## Model

The weights are not in this repository. They come from Meta's
[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
under the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).
Convert the checkpoint to an F16 GGUF with llama.cpp's `convert_hf_to_gguf.py`,
then quantize every matrix to Q4_0. Without the two tensor-type flags,
`output.weight` stays Q6_K and the engine cannot take it:

```bash
llama-quantize --output-tensor-type q4_0 --token-embedding-type q4_0 \
  llama-3.2-1b-instruct-f16.gguf llama-3.2-1b-instruct-q4_0-pure.gguf Q4_0
```

The file served here is 703,209,184 bytes, sha256
`b940c6c675634fb498a3cf44c0b301c04b6913062bf1c7d46ffcb6983c3f1b13`
([`0013-model-files.txt`](buildlog/evidence/0013-model-files.txt)).
