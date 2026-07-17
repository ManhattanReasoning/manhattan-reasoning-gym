"""Local synth-report smoke test for the plain-Verilog echo memory.

Unlike hello_wishbone/tests/ (Amaranth's own simulator, which only runs
Amaranth's IR), there's no Verilog simulator wired into mrg_build yet -- see
this example's README -- so this drives a real local synth report instead:
confirms the top module auto-detects, the design synthesizes, and the 512 x
32-bit memory infers block RAM, same as hello_wishbone's EchoSlave. Needs the
FPGA toolchain (in-process) or Docker; skips cleanly otherwise.
"""

from pathlib import Path

import pytest

import manhattan_reasoning_gym as mrg

_DESIGN = Path(__file__).resolve().parents[1] / "design.v"


def test_synth_infers_block_ram():
    try:
        rep = mrg.build.synth(_DESIGN)
    except mrg.build.SandboxUnavailableError as exc:
        pytest.skip(str(exc))
    assert rep.ok and rep.scope == "core"
    assert rep.synth_cells.get("DP16KD", 0) >= 1
