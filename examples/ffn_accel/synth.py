"""Synthesize FFN-accelerator blocks to the ECP5-85 and report utilization.

Emits Verilog with Amaranth, then runs yosys ``synth_ecp5`` (+ optional
nextpnr-ecp5 place-and-route) against the cloud board's part
(LFE5UM5G-85F, CABGA381) and prints the LUT4 / DSP / BRAM counts. Use it to see
how the parallel MAC grid scales and confirm it fits the 84k-LUT / 156-DSP part.

    python synth.py macarray --tile 16        # just yosys stat (fast)
    python synth.py macarray --tile 16 --pnr  # full place-and-route utilization
    python synth.py engine --m 8 --k 32 --n 8 --tile 8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from amaranth.back import verilog
from design import Gelu, MacArray, MatMulEngine, Requant

OUT = Path("/private/tmp/claude-501/ffn_accel_synth")
PART, PACKAGE, SPEED = "85k", "CABGA381", "8"


def _build(name: str, args) -> tuple[object, list]:
    if name == "macarray":
        dut = MacArray(args.tile, args.tile)
        ports = [dut.clear, dut.en, *dut.a_col, *dut.b_row, *dut.c]
    elif name == "requant":
        dut = Requant()
        ports = [dut.in_valid, dut.acc, dut.mult, dut.shift, dut.out_valid, dut.out]
    elif name == "gelu":
        dut = Gelu()
        ports = [
            dut.load_en,
            dut.load_addr,
            dut.load_data,
            dut.in_valid,
            dut.in_code,
            dut.out_valid,
            dut.out,
        ]
    elif name == "engine":
        dut = MatMulEngine(args.m, args.k, args.n, args.tile, args.tile)
        ports = [
            dut.a_we,
            dut.a_waddr,
            dut.a_wdata,
            dut.b_we,
            dut.b_waddr,
            dut.b_wdata,
            dut.mult,
            dut.shift,
            dut.start,
            dut.busy,
            dut.done,
            dut.c_raddr,
            dut.c_rdata,
        ]
    else:
        raise SystemExit(f"unknown block {name!r}")
    return dut, ports


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("block", choices=["macarray", "requant", "gelu", "engine"])
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--pnr", action="store_true", help="also run nextpnr-ecp5")
    args = p.parse_args()

    for tool in ["yosys"] + (["nextpnr-ecp5"] if args.pnr else []):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found on PATH (need oss-cad-suite)")

    OUT.mkdir(parents=True, exist_ok=True)
    dut, ports = _build(args.block, args)
    top = f"{args.block}_top"
    vpath = OUT / f"{top}.v"
    vpath.write_text(verilog.convert(dut, name=top, ports=ports))
    print(f"# wrote {vpath} ({vpath.stat().st_size} bytes)")

    json_out = OUT / f"{top}.json"
    ys = f"read_verilog {vpath}; synth_ecp5 -top {top} -json {json_out}; stat"
    subprocess.run(["yosys", "-q", "-p", ys], check=True)

    if args.pnr:
        print("\n# nextpnr-ecp5 place-and-route utilization:")
        subprocess.run(
            [
                "nextpnr-ecp5",
                f"--{PART}",
                "--package",
                PACKAGE,
                "--speed",
                SPEED,
                "--json",
                str(json_out),
                "--placer",
                "heap",
                "--router",
                "router2",
            ],
            check=True,
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
