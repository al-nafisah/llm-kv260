"""Contract v2, as executable checks.

Three things have to hold. The codec must read ggml's bytes the way ggml
writes them (checked against gguf-py, which is checked against ggml's C).
The GEMV must pin an order, observably. And a golden vector must freeze the
whole thing so a silent change to any step fails loudly.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from llm_kv260.q4_0 import LANES
from llm_kv260.q4_0 import QK
from llm_kv260.q4_0 import dequantize_q4_0
from llm_kv260.q4_0 import gemv_q4_0
from llm_kv260.q4_0 import gemv_q4_0_f32
from llm_kv260.q4_0 import pack_x_for_engine
from llm_kv260.q4_0 import quantize_q4_0
from llm_kv260.q4_0 import quantize_q8_0
from llm_kv260.q4_0 import unpack_q4_0

gguf = pytest.importorskip("gguf", reason="gguf-py provides the ggml-side reference")


def _weights(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, k)).astype(np.float32)


# ------------------------------------------------------------ codec vs ggml

def test_q4_0_bytes_match_gguf_py():
    w = _weights(16, 256, 0)
    ours = quantize_q4_0(w)
    theirs = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q4_0)
    assert np.array_equal(ours, theirs.reshape(ours.shape))


def test_q8_0_bytes_match_gguf_py():
    x = _weights(1, 8192, 1)[0]
    q, d = quantize_q8_0(x)
    theirs = gguf.quants.quantize(x.reshape(1, -1), gguf.GGMLQuantizationType.Q8_0)
    theirs = theirs.reshape(-1, 34)
    assert np.array_equal(d.view(np.uint8).reshape(-1, 2), theirs[:, :2])
    assert np.array_equal(q.view(np.uint8).reshape(-1, 32), theirs[:, 2:])


def test_q8_0_rounds_half_away_from_zero_like_roundf():
    """np.round would give half-to-even; ggml uses roundf. 2.5 -> 3, -2.5 -> -3."""
    x = np.zeros(QK, dtype=np.float32)
    x[0] = 127.0          # amax, so d = 1 and the others quantize to themselves
    x[1] = 2.5
    x[2] = -2.5
    x[3] = 0.5
    q, _ = quantize_q8_0(x)
    assert q[:4].tolist() == [127, 3, -3, 1]


def test_dequantize_matches_gguf_py():
    w = _weights(8, 512, 2)
    blocks = quantize_q4_0(w)
    theirs = gguf.quants.dequantize(blocks.reshape(-1), gguf.GGMLQuantizationType.Q4_0)
    assert np.array_equal(dequantize_q4_0(blocks, 512), theirs.reshape(8, 512))


def test_nibble_layout_is_low_then_high_half_block():
    """Byte j: low nibble = weight j, high nibble = weight j+16. Nibble n means n-8."""
    row = np.zeros((1, 18), dtype=np.uint8)
    row[0, :2] = np.array([1.0], dtype=np.float16).view(np.uint8)
    row[0, 2] = 0x0F | (0x00 << 4)     # weight 0 = 7, weight 16 = -8
    row[0, 3] = 0x08 | (0x09 << 4)     # weight 1 = 0, weight 17 = 1
    ints, d = unpack_q4_0(row, 32)
    assert d[0, 0] == np.float16(1.0)
    assert ints[0, 0] == 7 and ints[0, 16] == -8
    assert ints[0, 1] == 0 and ints[0, 17] == 1


def test_codec_keeps_fixed_width_dtypes():
    blocks = quantize_q4_0(_weights(2, 64, 3))
    ints, d = unpack_q4_0(blocks, 64)
    assert blocks.dtype == np.uint8 and ints.dtype == np.int8 and d.dtype == np.float16
    q, xd = quantize_q8_0(_weights(1, 64, 4)[0])
    assert q.dtype == np.int8 and xd.dtype == np.float16


def test_codec_rejects_wrong_inputs():
    with pytest.raises(TypeError):
        quantize_q4_0(np.zeros((2, 32), dtype=np.float64))
    with pytest.raises(TypeError):
        quantize_q8_0(np.zeros(32, dtype=np.float64))
    with pytest.raises(ValueError):
        quantize_q8_0(np.zeros(33, dtype=np.float32))
    with pytest.raises(ValueError):
        unpack_q4_0(np.zeros((1, 17), dtype=np.uint8), 32)


# ------------------------------------------------------------------ the gemv

def test_gemv_agrees_with_dequantized_matmul_to_float_precision():
    """Same answer as (dequant W) @ (dequant x), up to float rounding.

    This is a sanity check that the contract computes the right *thing*.
    The bit-exact checks below are about computing it in the right *order*.
    """
    w = _weights(64, 2048, 5)
    x = _weights(1, 2048, 6)[0]
    blocks = quantize_q4_0(w)
    xq, xd = quantize_q8_0(x)
    y = gemv_q4_0(blocks, 2048, xq, xd)
    x_deq = xq.astype(np.float32) * np.repeat(xd.astype(np.float32), QK)
    reference = dequantize_q4_0(blocks, 2048).astype(np.float64) @ x_deq.astype(np.float64)
    np.testing.assert_allclose(y, reference, rtol=1e-4, atol=1e-3)


def test_gemv_is_exact_when_only_one_lane_is_populated():
    """With nb == LANES every block lands in its own lane, and the tree is
    the whole reduction. Predict the result by hand, bit for bit."""
    k = QK * LANES
    w = _weights(4, k, 7)
    x = _weights(1, k, 8)[0]
    blocks = quantize_q4_0(w)
    xq, xd = quantize_q8_0(x)
    ints, dw = unpack_q4_0(blocks, k)
    sumi = (ints.reshape(4, LANES, QK).astype(np.int32)
            * xq.reshape(LANES, QK).astype(np.int32)).sum(-1, dtype=np.int32)
    t = (sumi.astype(np.float32) * dw.astype(np.float32)) * xd.astype(np.float32)
    # tree over 8 lanes: ((0+1)+(2+3)) + ((4+5)+(6+7))
    expected = ((t[:, 0] + t[:, 1]) + (t[:, 2] + t[:, 3])) + ((t[:, 4] + t[:, 5]) + (t[:, 6] + t[:, 7]))
    assert np.array_equal(gemv_q4_0(blocks, k, xq, xd), expected)


def test_lane_order_is_observable():
    """If the contract stops pinning the order, this test fails: the
    sequential-over-all-blocks order (llama.cpp's scalar path) gives
    different bits on real-sized inputs."""
    k = 2048
    w = _weights(256, k, 9)
    x = _weights(1, k, 10)[0]
    blocks = quantize_q4_0(w)
    xq, xd = quantize_q8_0(x)
    ours = gemv_q4_0(blocks, k, xq, xd)

    ints, dw = unpack_q4_0(blocks, k)
    nb = k // QK
    sumi = (ints.reshape(256, nb, QK).astype(np.int32)
            * xq.reshape(nb, QK).astype(np.int32)).sum(-1, dtype=np.int32)
    t = (sumi.astype(np.float32) * dw.astype(np.float32)) * xd.astype(np.float32)
    sequential = np.zeros(256, dtype=np.float32)
    for b in range(nb):
        sequential = sequential + t[:, b]

    differ = np.count_nonzero(ours.view(np.uint32) != sequential.view(np.uint32))
    assert differ > 0, "orders agreed bit-for-bit; the contract no longer pins anything"
    # Near-zero outputs cancel from O(1) terms, so compare against the terms' scale.
    np.testing.assert_allclose(ours, sequential, rtol=1e-4, atol=1e-3)


def test_gemv_output_is_float32_and_rejects_wrong_dtypes():
    blocks = quantize_q4_0(_weights(2, 256, 11))
    xq, xd = quantize_q8_0(_weights(1, 256, 12)[0])
    assert gemv_q4_0(blocks, 256, xq, xd).dtype == np.float32
    with pytest.raises(TypeError):
        gemv_q4_0(blocks, 256, xq.astype(np.int32), xd)
    with pytest.raises(ValueError):
        gemv_q4_0(quantize_q4_0(_weights(2, 32 * 4, 13)), 128, *quantize_q8_0(_weights(1, 128, 14)[0]))


def test_engine_x_layout():
    xq, xd = quantize_q8_0(_weights(1, 64, 15)[0])
    packed = pack_x_for_engine(xq, xd)
    assert packed.dtype == np.uint8 and packed.shape == (64 + 2 * 4,)
    assert np.array_equal(packed[:64].view(np.int8), xq)
    assert np.array_equal(packed[64:].view(np.float32), xd.astype(np.float32))


# ------------------------------------------------------------------- golden

GOLDEN_SHA256 = "efa9eda8cef689d6392ddf2e1b3e945d19797aff1cbd55398c87ffffec7c29cc"


def test_golden_vector():
    """Freeze the entire contract. Regenerate only with a version bump and a
    build-log entry saying why."""
    w = _weights(96, 2048, 2024)
    x = _weights(1, 2048, 2025)[0]
    y = gemv_q4_0_f32(quantize_q4_0(w), 2048, x)
    digest = hashlib.sha256(y.tobytes()).hexdigest()
    assert digest == GOLDEN_SHA256, digest
