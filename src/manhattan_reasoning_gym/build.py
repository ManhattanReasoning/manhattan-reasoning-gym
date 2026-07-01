"""Local build feedback — fast synth/PnR reports, no cloud and no board.

Auto-selects its backend: runs the FPGA toolchain **in-process** inside the
sandbox image, or transparently runs the **pinned Docker image** on a user's
machine. So ``pip install`` + Docker is all a user needs — they never install
yosys/nextpnr/LiteX. Raises ``SandboxUnavailableError`` if neither the toolchain
nor Docker is available.
"""

from __future__ import annotations

from ._local_build import SandboxUnavailableError, pnr, synth
from ._report import BuildReport, ResourceUse, Utilization

__all__ = [
    "synth",
    "pnr",
    "BuildReport",
    "Utilization",
    "ResourceUse",
    "SandboxUnavailableError",
]
