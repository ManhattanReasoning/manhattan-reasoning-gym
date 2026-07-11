"""End-to-end sim test: the whole FFN sublayer, bit-exact against the golden model.

Builds an FFN plan (calibrated int8 scales, requant constants, GELU table) with
:func:`golden.build_plan`, loads it into the hardware, runs the full
``requant2(GELU(requant1(x.W1)).W2)`` pipeline, and checks the int8 output
matches ``plan.y_q`` exactly -- then confirms the dequantized result tracks the
float-reference FFN within quantization tolerance.
"""

from __future__ import annotations

import golden
import numpy as np
from amaranth.sim import Simulator
from design import FFN

IN_W = 8


def _pack_rows(mat, lanes):  # [groups*lanes, depth] -> word[g*depth + d]
    groups, depth = mat.shape[0] // lanes, mat.shape[1]
    return [
        sum((int(mat[g * lanes + i, d]) & 0xFF) << (i * IN_W) for i in range(lanes))
        for g in range(groups)
        for d in range(depth)
    ]


def _pack_cols(mat, lanes):  # [depth, groups*lanes] -> word[d*groups + g]
    depth, groups = mat.shape[0], mat.shape[1] // lanes
    return [
        sum((int(mat[d, g * lanes + j]) & 0xFF) << (j * IN_W) for j in range(lanes))
        for d in range(depth)
        for g in range(groups)
    ]


def _s8(v):
    return v - 256 if v & 0x80 else v


def _run_ffn(plan, M, K1, K2, N, tile_m, tile_n):
    dut = FFN(M, K1, K2, N, tile_m, tile_n)
    xa = _pack_rows(plan.qx, tile_m)
    w1 = _pack_cols(plan.qW1, tile_n)
    w2 = _pack_cols(plan.qW2, tile_n)
    out = np.zeros((M, N), dtype=np.int64)

    async def load(ctx, we, waddr, wdata, words):
        for addr, w in enumerate(words):
            ctx.set(waddr, addr)
            ctx.set(wdata, w)
            ctx.set(we, 1)
            await ctx.tick()
        ctx.set(we, 0)

    async def tb(ctx):
        ctx.set(dut.mult1, plan.mult1)
        ctx.set(dut.shift1, plan.shift1)
        ctx.set(dut.mult2, plan.mult2)
        ctx.set(dut.shift2, plan.shift2)
        await load(ctx, dut.xa_we, dut.xa_waddr, dut.xa_wdata, xa)
        await load(ctx, dut.w1_we, dut.w1_waddr, dut.w1_wdata, w1)
        await load(ctx, dut.w2_we, dut.w2_waddr, dut.w2_wdata, w2)
        for addr in range(256):  # GELU table (loaded to all lanes in parallel)
            ctx.set(dut.g_load_addr, addr)
            ctx.set(dut.g_load_data, int(plan.lut[addr]))
            ctx.set(dut.g_load_en, 1)
            await ctx.tick()
        ctx.set(dut.g_load_en, 0)

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        for _ in range(500000):
            if ctx.get(dut.done):
                break
            await ctx.tick()
        else:
            raise AssertionError("FFN never asserted done")

        MTM = M // tile_m
        for mi in range(MTM):
            for col in range(N):
                ctx.set(dut.c_raddr, mi * N + col)
                await ctx.tick()
                w = ctx.get(dut.c_rdata)
                for lane in range(tile_m):
                    out[mi * tile_m + lane, col] = _s8((w >> (lane * IN_W)) & 0xFF)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    return out


def _build(M, K1, K2, N, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, K1))
    W1 = rng.standard_normal((K1, K2)) / np.sqrt(K1)
    W2 = rng.standard_normal((K2, N)) / np.sqrt(K2)
    return x, W1, W2, golden.build_plan(x, W1, W2)


def test_ffn_bit_exact_small():
    M, K1, K2, N = 4, 8, 8, 4
    x, W1, W2, plan = _build(M, K1, K2, N, seed=0)
    got = _run_ffn(plan, M, K1, K2, N, 4, 4)
    assert np.array_equal(got, plan.y_q), f"\n got=\n{got}\nwant=\n{plan.y_q}"


def test_ffn_bit_exact_tiled_and_accurate():
    # 4x hidden expansion (the BERT FFN shape), forcing multi-tile sweeps.
    M, K1, K2, N = 8, 8, 32, 8
    x, W1, W2, plan = _build(M, K1, K2, N, seed=3)
    got = _run_ffn(plan, M, K1, K2, N, 8, 8)
    assert np.array_equal(got, plan.y_q)

    # the dequantized hardware output tracks the exact float FFN
    block_float = golden.gelu(x @ W1) @ W2
    deq = got.astype(np.float64) * plan.s_y
    rel = np.abs(deq - block_float).max() / (np.abs(block_float).max() + 1e-9)
    assert rel < 0.05, f"quantization error too high: {rel:.3f}"
