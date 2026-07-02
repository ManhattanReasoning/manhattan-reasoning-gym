"""Local synth / PnR — fast, no-cloud, no-board build feedback.

Auto-selects a backend so the *same call* works everywhere:

  * inside the sandbox image (the FPGA toolchain is present) -> run in-process.
  * on a user's machine (toolchain absent)                   -> transparently
    run the pinned Docker image and parse its JSON report.

So a plain ``pip install`` user needs only Docker — they never install
yosys/nextpnr/LiteX. Either way a ``BuildReport`` comes back.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ._report import BuildReport

DEFAULT_SANDBOX_IMAGE = "ghcr.io/manhattanreasoning/mrg-sandbox:latest"


class SandboxUnavailableError(RuntimeError):
    """No way to run a local build: neither the toolchain nor Docker is available."""


def _image() -> str:
    return os.environ.get("MRG_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)


def _have_local_toolchain() -> bool:
    """True inside the sandbox image (mrg_build importable); False on a plain host."""
    try:
        import mrg_build  # noqa: F401
    except ImportError:
        return False
    return True


def synth(design: str | Path, *, work: str | Path | None = None) -> BuildReport:
    """Synthesis report for a user ``design.py``: resource util, fast, no timing."""
    return _build("synth", design, work=work)


def pnr(
    design: str | Path,
    *,
    target_mhz: float | None = None,
    sys_clk_mhz: float | None = None,
    timing_target_mhz: float | None = None,
    seed: int = 1,
    work: str | Path | None = None,
) -> BuildReport:
    """Full-SoC place-and-route report: Fmax, timing-met, SoC-wide util.

    ``sys_clk_mhz`` re-clocks the SoC (PLL output); ``timing_target_mhz`` is the
    constraint PnR optimizes against and ``timing_met`` is graded on, defaulting
    to the sys clock. ``target_mhz`` is a legacy alias that sets both — passing it
    alongside either new knob is rejected by the toolchain.
    """
    return _build(
        "pnr", design, target_mhz=target_mhz, sys_clk_mhz=sys_clk_mhz,
        timing_target_mhz=timing_target_mhz, seed=seed, work=work,
    )


def _build(
    mode, design, *, target_mhz=None, sys_clk_mhz=None, timing_target_mhz=None,
    seed=1, work=None,
) -> BuildReport:
    if _have_local_toolchain():
        return _build_in_process(
            mode, design, target_mhz, sys_clk_mhz, timing_target_mhz, seed, work
        )
    return _build_in_docker(mode, design, target_mhz, sys_clk_mhz, timing_target_mhz)


def _build_in_process(
    mode, design, target_mhz, sys_clk_mhz, timing_target_mhz, seed, work
) -> BuildReport:
    import mrg_build

    rep = mrg_build.build(
        mode=mode, design=design, target_mhz=target_mhz, sys_clk_mhz=sys_clk_mhz,
        timing_target_mhz=timing_target_mhz, seed=seed, work=work,
    )
    return BuildReport.from_dict(rep.to_dict())


def _build_in_docker(
    mode, design, target_mhz, sys_clk_mhz=None, timing_target_mhz=None
) -> BuildReport:
    if shutil.which("docker") is None:
        raise SandboxUnavailableError(
            "local synth/pnr needs the FPGA toolchain. Install Docker so mrg can "
            "run the pinned sandbox image, or run inside the mrg-sandbox image."
        )
    design = Path(design).resolve()
    if not design.exists():
        raise FileNotFoundError(f"design not found: {design}")

    cmd = [
        "docker", "run", "--rm", "--network=none",
        "-v", f"{design.parent}:/work:ro",
        _image(), "mrg", mode, f"/work/{design.name}",
    ]
    if target_mhz:
        cmd += ["--target-mhz", str(target_mhz)]
    # Only forward the new flags when explicitly set — a published image may
    # predate them, and the toolchain rejects mixing them with --target-mhz.
    if sys_clk_mhz:
        cmd += ["--sys-clk-mhz", str(sys_clk_mhz)]
    if timing_target_mhz:
        cmd += ["--timing-target-mhz", str(timing_target_mhz)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout.strip()
    if not out:
        raise SandboxUnavailableError(
            f"sandbox image {_image()!r} could not run (rc={proc.returncode}). "
            f"Is it built/pulled? {proc.stderr.strip()[-300:]}"
        )
    # `mrg synth/pnr` prints the JSON report even on a failed build (ok=False),
    # so parse stdout regardless of return code.
    return BuildReport.from_dict(json.loads(out))
