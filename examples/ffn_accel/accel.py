"""Host-side driver for the ffn_accel design over the SDK's MMIO window.

The reusable seam between a `manhattan_reasoning_gym.cloud.App` (wired to
`design.py`) and the FFN engine on the board: build a quantization plan with
:mod:`golden`, stream it through the windowed register map, run, and read the
int8 result back. Both this example's client and `examples/bert_ffn` drive the
accelerator through here, so the register contract lives in exactly one place.

The whole FFN sublayer -- `requant2(GELU(requant1(x.W1)).W2)` -- runs on-chip;
the host only quantizes, loads, and dequantizes.
"""

from __future__ import annotations

import golden
import manhattan_reasoning_gym as mrg
import numpy as np

# Baked-in hardware shape (mirrors design.py). 4x hidden expansion, 8->32->8.
M, K1, K2, N, TILE = 4, 8, 32, 8, 4
T_X, T_W1, T_W2, T_LUT = 0, 1, 2, 3


class Regs(mrg.cloud.RegisterMap):
    CTRL = 0x00  # W bit0=start; R bit0=done, bit1=busy
    TARGET = 0x04  # W buffer select (T_X / T_W1 / T_W2 / T_LUT)
    PTR = 0x08  # W base index into the selected buffer
    MULT1 = 0x0C
    SHIFT1 = 0x10
    MULT2 = 0x14
    SHIFT2 = 0x18
    ID_M = 0x1C
    ID_K2 = 0x24
    ID_TILE = 0x2C
    DATA = 0x40  # windowed buffer access (word 16)


def _pack_rows(mat, lanes):
    g, d = mat.shape[0] // lanes, mat.shape[1]
    return [
        sum((int(mat[gi * lanes + i, di]) & 0xFF) << (i * 8) for i in range(lanes))
        for gi in range(g)
        for di in range(d)
    ]


def _pack_cols(mat, lanes):
    d, g = mat.shape[0], mat.shape[1] // lanes
    return [
        sum((int(mat[di, gi * lanes + j]) & 0xFF) << (j * 8) for j in range(lanes))
        for di in range(d)
        for gi in range(g)
    ]


def _s8(v):
    return v - 256 if v & 0x80 else v


def check_dims(app) -> None:
    """Raise if the programmed hardware doesn't match this driver's shape."""
    if app.read(Regs.ID_M) != M or app.read(Regs.ID_K2) != K2:
        raise RuntimeError("hardware dims do not match the ffn_accel driver")


def _load(app, target, words):
    app.write(Regs.TARGET, target)
    app.write(Regs.PTR, 0)
    app.write(Regs.DATA, list(words))  # burst into buffer[0 ..]


def run_plan(app, plan) -> np.ndarray:
    """Stream a golden FFN plan to the board, run it, return the int8 output [M,N]."""
    _load(app, T_X, _pack_rows(plan.qx, TILE))
    _load(app, T_W1, _pack_cols(plan.qW1, TILE))
    _load(app, T_W2, _pack_cols(plan.qW2, TILE))
    _load(app, T_LUT, [int(v) & 0xFF for v in plan.lut])
    app.write(Regs.MULT1, int(plan.mult1))
    app.write(Regs.SHIFT1, int(plan.shift1))
    app.write(Regs.MULT2, int(plan.mult2))
    app.write(Regs.SHIFT2, int(plan.shift2))

    app.write(Regs.CTRL, 1)  # start
    while not (app.read(Regs.CTRL) & 1):  # poll done
        pass

    app.write(Regs.PTR, 0)
    words = app.read(Regs.DATA, count=(M // TILE) * N)
    out = np.zeros((M, N), dtype=np.int64)
    for mi in range(M // TILE):
        for col in range(N):
            w = words[mi * N + col]
            for lane in range(TILE):
                out[mi * TILE + lane, col] = _s8((w >> (lane * 8)) & 0xFF)
    return out


def _padded(x: np.ndarray) -> tuple[np.ndarray, int]:
    seq = x.shape[0]
    if seq == M:
        return x, seq
    return np.pad(x, ((0, M - seq), (0, 0))), seq


def run_ffn(app, x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Run the full FFN sublayer on the board; verify bit-exact; return float [seq,N].

    ``x`` may have fewer than M rows (it is zero-padded to the hardware's M and
    the result sliced back). The int8 hardware output is checked against the
    golden model and the dequantized result returned.
    """
    xp, seq = _padded(x)
    plan = golden.build_plan(xp, W1, W2)
    y_q = run_plan(app, plan)
    if not np.array_equal(y_q, plan.y_q):
        raise AssertionError(f"hardware FFN mismatch:\n got {y_q}\n want {plan.y_q}")
    return (y_q.astype(np.float64) * plan.s_y)[:seq]


def sim_ffn(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Hardware-free FFN backend: the golden model, bit-identical to the board."""
    xp, seq = _padded(x)
    plan = golden.build_plan(xp, W1, W2)
    return (plan.y_q.astype(np.float64) * plan.s_y)[:seq]
