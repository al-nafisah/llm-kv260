"""Golden vectors for the HLS testbench, straight from the contract.

Writes, per case, `w.bin` (the Q4_0 rows as ggml stores them), `x.bin` (the
engine's activation layout from `pack_x_for_engine`), `y.bin` (float32 from
`gemv_q4_0`) and `meta.txt` (k, rows). The testbench compares y byte for byte.

    .venv/bin/python hls/int4_core/gen_vectors.py hls/int4_core/vectors
"""

from __future__ import annotations

import os
import sys

import numpy as np

from llm_kv260.q4_0 import gemv_q4_0
from llm_kv260.q4_0 import pack_x_for_engine
from llm_kv260.q4_0 import quantize_q4_0
from llm_kv260.q4_0 import quantize_q8_0

# (k, rows): the n32 cases give each of gemv2's 8 MAC units a multiple of 4
# rows. The n16 cases cover the contract's own shape range; tb_gemv2 skips them.
CASES = [
    (256, 16),     # nb = 8: one chunk per row, every lane holds exactly one block
    (2048, 64),    # Llama-3.2-1B attention / gate / up widths
    (8192, 16),    # Llama-3.2-1B down_proj width, MAX_K
    (256, 32),
    (8192, 32),
    (512, 64),     # attn_output's width on one rank of four (o_k = 2048/4)
    (1024, 32),
]


def main(out_dir: str, cases=None) -> None:
    rng = np.random.default_rng(4)
    for k, rows in cases or CASES:
        case = os.path.join(out_dir, f"k{k}_n{rows}")
        os.makedirs(case, exist_ok=True)
        w = rng.standard_normal((rows, k)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        blocks = quantize_q4_0(w)
        xq, xd = quantize_q8_0(x)
        y = gemv_q4_0(blocks, k, xq, xd)
        blocks.tofile(os.path.join(case, "w.bin"))
        pack_x_for_engine(xq, xd).tofile(os.path.join(case, "x.bin"))
        y.tofile(os.path.join(case, "y.bin"))
        with open(os.path.join(case, "meta.txt"), "w") as handle:
            handle.write(f"{k} {rows}\n")
        print(f"{case}: w {blocks.nbytes} B, x {k + k // 8} B, y {y.nbytes} B")


if __name__ == "__main__":
    # gen_vectors.py DIR [k,rows ...]: extra shapes on demand, e.g. one rank's shard shapes
    extra = [tuple(int(v) for v in a.split(",")) for a in sys.argv[2:]]
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "vectors"), extra or None)
