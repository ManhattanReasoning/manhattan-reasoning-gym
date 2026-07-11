"""Sim test: the MAC grid accumulates a full int8 tile dot-product bit-exactly.

Drives the clear/en contract from :mod:`rtl.mac_array` -- one clear cycle, then
K enabled cycles presenting column k of A and row k of B -- and checks every
accumulator against an integer NumPy reference. No hardware required.
"""

from __future__ import annotations

import numpy as np
from amaranth.sim import Simulator
from design import MacArray


def _run_tile(a: np.ndarray, b: np.ndarray, tile_m: int, tile_n: int) -> np.ndarray:
    """Simulate one tile through MacArray; return the [tile_m, tile_n] result."""
    k = a.shape[1]
    dut = MacArray(tile_m, tile_n)
    out = np.zeros((tile_m, tile_n), dtype=np.int64)

    async def tb(ctx):
        ctx.set(dut.clear, 1)
        await ctx.tick()
        ctx.set(dut.clear, 0)
        for kk in range(k):
            for i in range(tile_m):
                ctx.set(dut.a_col[i], int(a[i, kk]))
            for j in range(tile_n):
                ctx.set(dut.b_row[j], int(b[kk, j]))
            ctx.set(dut.en, 1)
            await ctx.tick()
        ctx.set(dut.en, 0)
        await ctx.tick()  # let the final accumulate settle before reading
        for i in range(tile_m):
            for j in range(tile_n):
                out[i, j] = ctx.get(dut.c[i * tile_n + j])

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    return out


def test_mac_array_matches_integer_matmul():
    rng = np.random.default_rng(0)
    tile_m, tile_n, k = 8, 8, 37
    a = rng.integers(-127, 128, size=(tile_m, k), dtype=np.int64)
    b = rng.integers(-127, 128, size=(k, tile_n), dtype=np.int64)
    got = _run_tile(a, b, tile_m, tile_n)
    want = a @ b
    assert np.array_equal(got, want), f"\n got=\n{got}\nwant=\n{want}"


def test_mac_array_handles_extremes_and_rectangular():
    # Saturated operands and a non-square tile exercise sign + width handling.
    tile_m, tile_n, k = 4, 6, 64
    a = np.full((tile_m, k), -127, dtype=np.int64)
    b = np.full((k, tile_n), 127, dtype=np.int64)
    got = _run_tile(a, b, tile_m, tile_n)
    want = a @ b
    assert np.array_equal(got, want)
