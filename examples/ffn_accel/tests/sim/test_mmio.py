"""Sim test: the full FFN runs through the Wishbone MMIO slave, bit-exact.

Drives exactly the transaction sequence the SDK client performs over the SoC's
2 KB window -- windowed operand loads, requant constants, start, poll done,
windowed result read -- against the simulated FFNSlave, and checks the int8
output against golden.build_plan. This is the live-board data path, in sim.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import golden
import numpy as np
from amaranth.sim import Simulator

_spec = importlib.util.spec_from_file_location(
    "ffn_design", Path(__file__).resolve().parents[2] / "design.py"
)
design = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(design)

FFNSlave = design.FFNSlave
M, K1, K2, N, TILE = design.M, design.K1, design.K2, design.N, design.TILE


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


def test_mmio_ffn_bit_exact():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, K1))
    W1 = rng.standard_normal((K1, K2)) / np.sqrt(K1)
    W2 = rng.standard_normal((K2, N)) / np.sqrt(K2)
    plan = golden.build_plan(x, W1, W2)

    dut = FFNSlave()
    out = np.zeros((M, N), dtype=np.int64)

    async def wb_write(ctx, adr, dat):
        ctx.set(dut.wb_cyc, 1)
        ctx.set(dut.wb_stb, 1)
        ctx.set(dut.wb_we, 1)
        ctx.set(dut.wb_adr, adr)
        ctx.set(dut.wb_dat_w, dat & 0xFFFFFFFF)
        for _ in range(8):
            await ctx.tick()
            if ctx.get(dut.wb_ack):
                break
        ctx.set(dut.wb_cyc, 0)
        ctx.set(dut.wb_stb, 0)
        ctx.set(dut.wb_we, 0)
        await ctx.tick()

    async def wb_read(ctx, adr):
        ctx.set(dut.wb_cyc, 1)
        ctx.set(dut.wb_stb, 1)
        ctx.set(dut.wb_we, 0)
        ctx.set(dut.wb_adr, adr)
        val = 0
        for _ in range(8):
            await ctx.tick()
            if ctx.get(dut.wb_ack):
                val = ctx.get(dut.wb_dat_r)
                break
        ctx.set(dut.wb_cyc, 0)
        ctx.set(dut.wb_stb, 0)
        await ctx.tick()
        return val

    async def load(ctx, target, words):
        await wb_write(ctx, design.TARGET, target)
        await wb_write(ctx, design.PTR, 0)
        for i, w in enumerate(words):
            await wb_write(ctx, design.DATA_WORD + i, w)

    async def tb(ctx):
        # verify the hardware shape matches the client's expectation
        assert await wb_read(ctx, design.ID_M) == M
        assert await wb_read(ctx, design.ID_K2) == K2

        await load(ctx, design.T_X, _pack_rows(plan.qx, TILE))
        await load(ctx, design.T_W1, _pack_cols(plan.qW1, TILE))
        await load(ctx, design.T_W2, _pack_cols(plan.qW2, TILE))
        await load(ctx, design.T_LUT, [int(v) & 0xFF for v in plan.lut])

        await wb_write(ctx, design.MULT1, plan.mult1)
        await wb_write(ctx, design.SHIFT1, plan.shift1)
        await wb_write(ctx, design.MULT2, plan.mult2)
        await wb_write(ctx, design.SHIFT2, plan.shift2)

        await wb_write(ctx, design.CTRL, 1)  # start
        for _ in range(500000):
            if await wb_read(ctx, design.CTRL) & 1:  # done
                break
        else:
            raise AssertionError("FFN never reported done over MMIO")

        await wb_write(ctx, design.PTR, 0)
        MTM = M // TILE
        for mi in range(MTM):
            for col in range(N):
                w = await wb_read(ctx, design.DATA_WORD + mi * N + col)
                for lane in range(TILE):
                    out[mi * TILE + lane, col] = _s8((w >> (lane * 8)) & 0xFF)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()

    assert np.array_equal(out, plan.y_q), f"\n got=\n{out}\nwant=\n{plan.y_q}"
