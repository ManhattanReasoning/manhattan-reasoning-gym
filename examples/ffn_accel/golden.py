"""Integer golden model for the streaming INT8 FFN accelerator.

Every operation here is the *exact* integer computation the silicon performs,
so a passing sim test means bit-exactness, not "close enough". The float world
appears only at the edges: choosing the int8 scales (calibration) and the
fixed-point requantization multipliers. Once those constants are chosen they
are integers, handed to the hardware, and reproduced cycle-for-cycle.

Pipeline (BERT FFN, ``GELU(x.W1).W2`` -- biases are zero in this demo, matching
``examples/bert_ffn/model.py``)::

    qx ─▶ matmul(qx, qW1) ─▶ requant1 ─▶ GELU-LUT ─▶ matmul(·, qW2) ─▶ requant2 ─▶ qy
          int32 acc          int8          int8        int32 acc        int8

The accelerator never sees a float. The golden model picks the constants; the
hardware consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mirror of design.py's fixed-point parameters (kept here so the golden model is
# pure-numpy and importable without Amaranth).
INT8_MAX = 127
REQUANT_MULT_W = 32


# ── int8 quantization ─────────────────────────────────────────────────────────


def quantize_int8(mat: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quantization: returns (q in [-127,127], scale)."""
    amax = float(np.abs(mat).max())
    scale = amax / INT8_MAX if amax > 0.0 else 1.0
    q = np.clip(np.round(mat / scale), -INT8_MAX, INT8_MAX).astype(np.int64)
    return q, scale


def scale_for(mat: np.ndarray) -> float:
    """The symmetric int8 scale a tensor *would* get, without quantizing it."""
    amax = float(np.abs(mat).max())
    return amax / INT8_MAX if amax > 0.0 else 1.0


# ── fixed-point requantization (int32 acc -> int8) ────────────────────────────


def choose_requant(real_mult: float) -> tuple[int, int]:
    """Represent a positive real multiplier as (mult, shift), gemmlowp-style.

    Picks ``shift`` so ``mult = round(real_mult * 2**shift)`` lands near 2**31,
    extracting maximum precision from a ``REQUANT_MULT_W``-bit integer. The
    hardware then computes ``(acc * mult + (1 << (shift-1))) >> shift``.
    """
    if real_mult <= 0.0:
        return 0, 1
    shift = REQUANT_MULT_W - 1 - int(np.floor(np.log2(real_mult)))
    mult = int(round(real_mult * (2.0**shift)))
    if mult >= (1 << REQUANT_MULT_W):  # rounded up past the width -> renormalize
        mult >>= 1
        shift -= 1
    return mult, shift


def requant(acc: np.ndarray, mult: int, shift: int) -> np.ndarray:
    """Apply ``clamp((acc*mult + round_bias) >> shift, -127, 127)`` in int64.

    Arithmetic (floor) right-shift on signed values, with a positive rounding
    bias of half an LSB -- the identical sequence the RTL performs, so results
    match bit-for-bit including the rounding of negative accumulators.
    """
    acc = acc.astype(np.int64)
    bias = 1 << (shift - 1)
    shifted = (acc * mult + bias) >> shift  # Python >> is arithmetic floor-shift
    return np.clip(shifted, -INT8_MAX, INT8_MAX).astype(np.int64)


# ── GELU as an int8 -> int8 lookup ────────────────────────────────────────────


def gelu(x: np.ndarray) -> np.ndarray:
    """BERT's tanh-approximation GELU (float)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_lut(s_in: float, s_out: float) -> np.ndarray:
    """256-entry int8->int8 GELU table, indexed by the two's-complement input.

    ``table[q & 0xff]`` is the int8 GELU of input code ``q`` (q in [-128,127]),
    where the input dequantizes with ``s_in`` and the output requantizes with
    ``s_out``. Exact over the entire int8 domain -- so the hardware lookup is
    not an approximation of GELU's *table*, it is the table.
    """
    codes = np.arange(-128, 128, dtype=np.int64)
    real = gelu(codes * s_in)
    q = np.clip(np.round(real / s_out), -INT8_MAX, INT8_MAX).astype(np.int64)
    table = np.empty(256, dtype=np.int64)
    table[codes & 0xFF] = q  # index by unsigned 8-bit two's-complement code
    return table


# ── integer matmul (tiler reference) ──────────────────────────────────────────


def int_matmul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Plain integer matmul ``[M,K]·[K,N] -> int32 [M,N]`` (int64 math, fits i32)."""
    return (qa.astype(np.int64) @ qb.astype(np.int64)).astype(np.int64)


# ── the full FFN plan ─────────────────────────────────────────────────────────


@dataclass
class FFNPlan:
    """Everything the hardware needs, plus the int8 output it must reproduce."""

    qx: np.ndarray  # [M, K1]  int8 activations
    qW1: np.ndarray  # [K1, K2] int8 weights (first projection)
    qW2: np.ndarray  # [K2, N]  int8 weights (second projection)
    mult1: int
    shift1: int  # requant after matmul-1
    lut: np.ndarray  # [256]   int8 GELU table
    mult2: int
    shift2: int  # requant after matmul-2
    y_q: np.ndarray  # [M, N]   expected int8 output (the bit-exact target)
    s_y: float  # output scale, for dequantizing y_q back to float


def build_plan(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> FFNPlan:
    """Calibrate scales on ``x`` and produce the integer FFN plan + golden output.

    Mirrors static post-training quantization: float reference activations pick
    the int8 scales for each stage, which fix the requant multipliers and the
    GELU table. Returns the plan and the exact int8 output the silicon must hit.
    """
    qx, sx = quantize_int8(x)
    qW1, sw1 = quantize_int8(W1)
    qW2, sw2 = quantize_int8(W2)

    # Stage 1: matmul -> requant to int8 hidden. Calibrate s_h on the real product.
    acc1 = int_matmul(qx, qW1)
    real_hidden = x @ W1
    s_h = scale_for(real_hidden)
    mult1, shift1 = choose_requant((sx * sw1) / s_h)
    h_q = requant(acc1, mult1, shift1)

    # GELU table: maps the int8 hidden (scale s_h) to int8 (scale s_g).
    s_g = scale_for(gelu(real_hidden))
    lut = gelu_lut(s_h, s_g)
    g_q = lut[h_q & 0xFF]

    # Stage 2: matmul -> requant to int8 output. Calibrate s_y on the real output.
    acc2 = int_matmul(g_q, qW2)
    real_out = gelu(real_hidden) @ W2
    s_y = scale_for(real_out)
    mult2, shift2 = choose_requant((s_g * sw2) / s_y)
    y_q = requant(acc2, mult2, shift2)

    return FFNPlan(qx, qW1, qW2, mult1, shift1, lut, mult2, shift2, y_q, s_y)
