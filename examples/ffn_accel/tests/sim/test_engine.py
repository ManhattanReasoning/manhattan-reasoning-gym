"""Sim test: the tiled engine computes requant(A@B) bit-exactly for full matrices.

Loads packed operands into the engine's BRAMs, pulses start, waits for done, and
reads the packed int8 result back -- checking it against the golden integer
reference over several non-trivial M/K/N shapes that force real tiling.
"""

from __future__ import annotations

import golden
import numpy as np
from amaranth.sim import Simulator
from design import MatMulEngine

IN_W = 8


def _pack_rows(mat: np.ndarray, lanes: int) -> list[int]:
    """Pack a [groups*lanes, depth] (A-style) or per-row int8 matrix into words.

    Returns words where word[g*depth + d] packs column d of group g's `lanes`
    rows: lane i (i=0..lanes-1) in bits [i*8, i*8+8).
    """
    groups = mat.shape[0] // lanes
    depth = mat.shape[1]
    words = []
    for g in range(groups):
        for d in range(depth):
            w = 0
            for i in range(lanes):
                w |= (int(mat[g * lanes + i, d]) & 0xFF) << (i * IN_W)
            words.append(w)
    return words


def _pack_cols(mat: np.ndarray, lanes: int) -> list[int]:
    """Pack a [depth, groups*lanes] (B-style) matrix: word[d*groups + g]."""
    depth = mat.shape[0]
    groups = mat.shape[1] // lanes
    words = []
    for d in range(depth):
        for g in range(groups):
            w = 0
            for j in range(lanes):
                w |= (int(mat[d, g * lanes + j]) & 0xFF) << (j * IN_W)
            words.append(w)
    return words


def _to_signed8(v: int) -> int:
    return v - 256 if v & 0x80 else v


def _run_engine(a, b, mult, shift, tile_m, tile_n):
    M, K = a.shape
    K2, N = b.shape
    dut = MatMulEngine(M, K, N, tile_m, tile_n)
    a_words = _pack_rows(a, tile_m)
    b_words = _pack_cols(b, tile_n)
    MTM = M // tile_m
    out = np.zeros((M, N), dtype=np.int64)

    async def tb(ctx):
        ctx.set(dut.mult, mult)
        ctx.set(dut.shift, shift)
        for addr, w in enumerate(a_words):
            ctx.set(dut.a_waddr, addr)
            ctx.set(dut.a_wdata, w)
            ctx.set(dut.a_we, 1)
            await ctx.tick()
        ctx.set(dut.a_we, 0)
        for addr, w in enumerate(b_words):
            ctx.set(dut.b_waddr, addr)
            ctx.set(dut.b_wdata, w)
            ctx.set(dut.b_we, 1)
            await ctx.tick()
        ctx.set(dut.b_we, 0)

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        for _ in range(200000):
            if ctx.get(dut.done):
                break
            await ctx.tick()
        else:
            raise AssertionError("engine never asserted done")

        # C is column-packed (A-layout): word[mi*N + col] holds tile_m row lanes.
        for mi in range(MTM):
            for col in range(N):
                ctx.set(dut.c_raddr, mi * N + col)
                await ctx.tick()  # 1-cycle BRAM read latency
                w = ctx.get(dut.c_rdata)
                for lane in range(tile_m):
                    out[mi * tile_m + lane, col] = _to_signed8(
                        (w >> (lane * IN_W)) & 0xFF
                    )

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    return out


def _check(M, K, N, tile_m, tile_n, real_mult, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, size=(M, K), dtype=np.int64)
    b = rng.integers(-127, 128, size=(K, N), dtype=np.int64)
    mult, shift = golden.choose_requant(real_mult)
    want = golden.requant(golden.int_matmul(a, b), mult, shift)
    got = _run_engine(a, b, mult, shift, tile_m, tile_n)
    assert np.array_equal(got, want), (
        f"shape {M}x{K}x{N} tile {tile_m}x{tile_n}\n got=\n{got}\nwant=\n{want}"
    )


def test_single_tile():
    _check(4, 5, 4, 4, 4, 0.02, seed=0)


def test_multi_tile_square():
    _check(8, 16, 8, 4, 4, 0.005, seed=1)


def test_rectangular_tiled():
    _check(16, 12, 24, 8, 8, 0.001, seed=2)
