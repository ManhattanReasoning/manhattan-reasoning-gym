"""The BuildReport the SDK hands back from ``mrg.build.synth`` / ``pnr``.

The JSON emitted by the toolchain (``mrg_build``) is the contract; this is the
SDK-owned typed view of it, so the return type is the same whether the build ran
in-process (in the sandbox image) or via the Docker backend on a user's machine —
and the SDK never has to import the image-only ``mrg_build``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class ResourceUse:
    used: int = 0
    available: int = 0

    @property
    def pct(self) -> float:
        return round(100.0 * self.used / self.available, 2) if self.available else 0.0


@dataclass
class Utilization:
    logic: ResourceUse
    ff: ResourceUse
    bram: ResourceUse
    dsp: ResourceUse


@dataclass
class BuildReport:
    mode: str  # "synth" | "pnr"
    ok: bool
    scope: str = "core"  # "core" | "soc"
    fits: bool | None = None
    fmax_mhz: float | None = None
    sys_clk_mhz: float | None = None
    target_mhz: float | None = None
    timing_met: bool | None = None
    clock: str | None = None
    util: Utilization | None = None
    synth_cells: dict[str, int] | None = None
    warnings: list[str] = field(default_factory=list)
    design_hash: str | None = None
    toolchain: str | None = None
    log_tail: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> BuildReport:
        u = d.get("util") or {}
        util = None
        if u:
            classes = ("logic", "ff", "bram", "dsp")
            util = Utilization(**{k: ResourceUse(**u.get(k, {})) for k in classes})
        return cls(
            mode=d["mode"], ok=d["ok"], scope=d.get("scope", "core"),
            fits=d.get("fits"), fmax_mhz=d.get("fmax_mhz"),
            sys_clk_mhz=d.get("sys_clk_mhz"),
            target_mhz=d.get("target_mhz"), timing_met=d.get("timing_met"),
            clock=d.get("clock"), util=util, synth_cells=d.get("synth_cells"),
            warnings=d.get("warnings", []), design_hash=d.get("design_hash"),
            toolchain=d.get("toolchain"), log_tail=d.get("log_tail"),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
