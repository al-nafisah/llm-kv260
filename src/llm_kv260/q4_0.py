"""The contract: the W4A8 GEMV the fabric engine computes, down to the bit.

The engine (hls/gemv2, on the datapath in hls/int4_core) is held to this module. A single differing byte
between what the fabric writes and what this computes is a failure.

## Why the engine consumes ggml's Q4_0 and Q8_0

Two reasons, both about tokens per second.

1. Integer accumulation is associative. With int8 activations, the 32 products
   inside a block are summed in int32, where every order gives the same bits.
   The only floating-point work left is one scale multiply per block and the
   accumulation *across* blocks. A fabric that multiplies float32 activations
   needs ~600 DSPs to keep up with DRAM; one that multiplies int4 by int8
   needs ~60. That is the difference between a bandwidth-bound engine and a
   compute-bound one.
2. Q4_0 is what a GGUF file already stores. The engine reads the model's own
   bytes in place -- no reformatting, no second copy of 600 MB.

## The formats (ggml-common.h)

Q4_0 block: 18 bytes = float16 `d`, then 16 bytes of nibbles for 32 weights.
Byte j holds weight j in its low nibble and weight j+16 in its high nibble.
A nibble n means the integer n - 8, so the range is [-8, 7].

Q8_0 block: 34 bytes = float16 `d`, then 32 int8 values.

The quantization rules are ggml's reference ones (`ggml-quants.c`) and are
reproduced here exactly, including `max / -8` for Q4_0 (the signed extreme,
so that the extreme value lands on -8) and `roundf` for Q8_0.

## The order the fabric produces, which this module pins

For one output row of K inputs (nb = K/32 blocks):

1. `sumi[b] = sum_j int(w[b, j]) * int(x[b, j])` in int32. Order-free.
2. `t[b] = (float32(sumi[b]) * float32(d_w[b])) * float32(d_x[b])`, two
   float32 multiplies, left to right. float16 -> float32 is exact.
3. Block b is added into lane `b mod 8`, lanes accumulate sequentially in
   increasing b. Eight lanes because a float32 add takes several cycles and
   the engine consumes one block per cycle: the dependence distance has to be
   longer than the adder latency, or the engine stalls to a quarter of DRAM
   bandwidth. Eight is the smallest power of two that clears it with margin.
4. The eight lanes are summed by a binary tree over adjacent pairs
   (`tree_sum`).

This is *not* the order llama.cpp's scalar reference uses (it accumulates
blocks sequentially into one float), so the engine will not bit-match a CPU
run. It is not meant to; the engine is held to *this* module, and the CPU is
the sanity check that the answers agree to float precision.

Everything is fixed-width NumPy. No Python ints, no float64.
"""

from __future__ import annotations

import numpy as np

QK = 32            # weights per block, shared by Q4_0 and Q8_0
Q4_0_BYTES = 18    # 2 (float16 scale) + 16 (nibbles)
Q8_0_BYTES = 34    # 2 (float16 scale) + 32 (int8)
LANES = 8

CONTRACT_VERSION = "llm-kv260/w4a8-q4_0-v2"


def tree_sum(values: np.ndarray) -> np.ndarray:
    """Sum the last axis by a binary tree over adjacent pairs.

    The last axis must be a power of two. Stays in the input dtype: no
    accumulator widening, because the fabric will not widen either.
    """
    length = values.shape[-1]
    if length == 0 or (length & (length - 1)):
        raise ValueError(f"reduction length must be a power of two, got {length}")
    while values.shape[-1] > 1:
        values = values[..., 0::2] + values[..., 1::2]
    return values[..., 0]


# ---------------------------------------------------------------- quantizers

def quantize_q8_0(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml's `quantize_row_q8_0_ref`. Returns (q int8 [K], d float16 [K/32])."""
    if x.dtype != np.float32:
        raise TypeError(f"activations must be float32, got {x.dtype}")
    if x.ndim != 1 or x.shape[0] % QK:
        raise ValueError(f"x must be 1-D with a multiple of {QK} elements, got {x.shape}")
    blocks = x.reshape(-1, QK)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    # roundf: half away from zero. np.round is half to even, so do it by hand.
    scaled = blocks * inv
    q = np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))
    return q.astype(np.int8).reshape(-1), d.astype(np.float16).reshape(-1)


def quantize_q4_0(w: np.ndarray) -> np.ndarray:
    """ggml's `quantize_row_q4_0_ref`. Returns the block bytes, uint8 [N, K/32*18].

    Real weights come out of a GGUF file, not this function. It exists so tests
    can build a matrix with known bytes and so `dequantize_q4_0` has an inverse
    to be checked against.
    """
    if w.dtype != np.float32:
        raise TypeError(f"weights must be float32, got {w.dtype}")
    n, k = w.shape
    if k % QK:
        raise ValueError(f"in_features must be a multiple of {QK}, got {k}")
    blocks = w.reshape(n, k // QK, QK)
    imax = np.abs(blocks).argmax(axis=-1, keepdims=True)
    extreme = np.take_along_axis(blocks, imax, axis=-1)
    d = (extreme / np.float32(-8)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    q = np.trunc(blocks * inv + np.float32(8.5)).astype(np.uint8).clip(0, 15)
    packed = q[..., :QK // 2] | (q[..., QK // 2:] << np.uint8(4))
    d16 = d.astype(np.float16).reshape(n, k // QK, 1).view(np.uint8)
    return np.concatenate([d16, packed], axis=-1).reshape(n, -1)


# -------------------------------------------------------------------- codec

def unpack_q4_0(blocks: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Split Q4_0 row bytes into (ints int8 [N, K] in [-8, 7], d float16 [N, K/32])."""
    if blocks.dtype != np.uint8:
        raise TypeError(f"expected uint8 block bytes, got {blocks.dtype}")
    n = blocks.shape[0]
    nb = k // QK
    if blocks.shape[1] != nb * Q4_0_BYTES:
        raise ValueError(f"{blocks.shape[1]} bytes per row is not {nb} blocks of {Q4_0_BYTES}")
    rows = blocks.reshape(n, nb, Q4_0_BYTES)
    d = rows[:, :, :2].copy().view(np.float16).reshape(n, nb)
    qs = rows[:, :, 2:]
    ints = np.empty((n, nb, QK), dtype=np.int8)
    ints[:, :, :QK // 2] = (qs & np.uint8(0x0F)).astype(np.int8) - np.int8(8)
    ints[:, :, QK // 2:] = (qs >> np.uint8(4)).astype(np.int8) - np.int8(8)
    return ints.reshape(n, k), d


def dequantize_q4_0(blocks: np.ndarray, k: int) -> np.ndarray:
    """Expand to float32 [N, K], the way gguf-py's `Q4_0.dequantize_blocks` does."""
    ints, d = unpack_q4_0(blocks, k)
    return ints.astype(np.float32) * np.repeat(d.astype(np.float32), QK, axis=1)


# --------------------------------------------------------------------- gemv

def gemv_q4_0(blocks: np.ndarray, k: int, xq: np.ndarray, xd: np.ndarray) -> np.ndarray:
    """y = W x with W in Q4_0 bytes and x in Q8_0, in the engine's order.

    `blocks` uint8   [N, K/32*18]   the rows exactly as stored in the GGUF
    `xq`     int8    [K]
    `xd`     float16 [K/32]

    Returns float32 [N].
    """
    if xq.dtype != np.int8 or xd.dtype != np.float16:
        raise TypeError(f"x must be (int8, float16), got ({xq.dtype}, {xd.dtype})")
    ints, dw = unpack_q4_0(blocks, k)
    n, nb = dw.shape
    if xq.shape != (k,) or xd.shape != (nb,):
        raise ValueError(f"x shapes {xq.shape}, {xd.shape} do not match K={k}")
    if nb % LANES:
        raise ValueError(f"{nb} blocks is not a multiple of {LANES} lanes")

    # Step 1: int32 block dot products. [N, nb]
    sumi = (ints.reshape(n, nb, QK).astype(np.int32)
            * xq.reshape(nb, QK).astype(np.int32)[np.newaxis]).sum(axis=-1, dtype=np.int32)

    # Step 2: two float32 multiplies, left to right. [N, nb]
    t = (sumi.astype(np.float32) * dw.astype(np.float32)) * xd.astype(np.float32)[np.newaxis]

    # Step 3: eight lanes, each a sequential accumulation over its blocks.
    lanes = np.zeros((n, LANES), dtype=np.float32)
    for b in range(nb):
        lanes[:, b % LANES] = lanes[:, b % LANES] + t[:, b]

    # Step 4: binary tree over the lanes.
    return tree_sum(lanes)


def gemv_q4_0_f32(blocks: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """Convenience: quantize float32 `x` to Q8_0 first, then `gemv_q4_0`."""
    xq, xd = quantize_q8_0(x)
    return gemv_q4_0(blocks, k, xq, xd)


# ------------------------------------------------- what the engine is fed

def pack_x_for_engine(xq: np.ndarray, xd: np.ndarray) -> np.ndarray:
    """The activation buffer layout the fabric reads: int8[K] then float32[K/32].

    The float16 -> float32 conversion is done here, on the PS, because it is
    exact either way and a float32 read is simpler in the fabric than a
    half-precision converter per master.
    """
    return np.concatenate([xq.view(np.uint8), xd.astype(np.float32).view(np.uint8)])
