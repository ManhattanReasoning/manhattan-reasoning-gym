"""Sim test: the hardware requantizer matches golden.requant bit-for-bit.

Sweeps real multipliers and a wide range of signed accumulators (including the
negatives whose rounding is easy to get wrong) and checks the int8 output
against the golden integer reference.
"""

from __future__ import annotations

import golden
import numpy as np
from amaranth.sim import Simulator
from design import Requant


def _run(accs: list[int], mult: int, shift: int) -> list[int]:
    dut = Requant()
    out: list[int] = []
    n = len(accs)

    async def tb(ctx):
        ctx.set(dut.mult, mult)
        ctx.set(dut.shift, shift)
        # Feed for n cycles, then 2 drain cycles; sample the 1-cycle-late output
        # at the top of each iteration (ctx.get returns signed ints directly).
        for idx in range(n + 2):
            if ctx.get(dut.out_valid):
                out.append(ctx.get(dut.out))
            if idx < n:
                ctx.set(dut.acc, int(accs[idx]))
                ctx.set(dut.in_valid, 1)
            else:
                ctx.set(dut.in_valid, 0)
            await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    return out


def test_requant_matches_golden():
    rng = np.random.default_rng(1)
    accs = [int(x) for x in rng.integers(-(2**24), 2**24, size=200)]
    accs += [0, 1, -1, 2**24, -(2**24)]  # edges
    for real_mult in [0.5, 0.123, 0.0009, 1.7, 3.9]:
        mult, shift = golden.choose_requant(real_mult)
        want = golden.requant(np.array(accs), mult, shift).tolist()
        got = _run(accs, mult, shift)
        assert got == want, f"real_mult={real_mult} mult={mult} shift={shift}"
