"""Unit tests for mrg.build's auto-backend (in-process vs Docker).

No toolchain or Docker is actually used — the backends are faked to assert the
selection logic and that both return a BuildReport.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from manhattan_reasoning_gym import _local_build
from manhattan_reasoning_gym._report import BuildReport

_REPORT = {
    "mode": "pnr", "ok": True, "scope": "soc", "fits": True,
    "fmax_mhz": 110.0, "timing_met": True,
    "util": {
        "logic": {"used": 10, "available": 100}, "ff": {"used": 5, "available": 100},
        "bram": {"used": 0, "available": 10}, "dsp": {"used": 1, "available": 10},
    },
    "design_hash": "sha256:abc",
}


def test_in_process_backend_when_toolchain_present(monkeypatch):
    """mrg_build importable => run in-process, return a BuildReport."""
    calls = {}
    fake = types.ModuleType("mrg_build")

    class _Rep:
        def to_dict(self):
            return _REPORT

    def build(**kwargs):
        calls.update(kwargs)
        return _Rep()

    fake.build = build
    monkeypatch.setitem(sys.modules, "mrg_build", fake)

    rep = _local_build.synth("design.py")
    assert isinstance(rep, BuildReport)
    assert rep.ok and rep.util.dsp.used == 1 and rep.util.logic.pct == 10.0
    assert calls["mode"] == "synth" and calls["design"] == "design.py"


def test_docker_backend_when_toolchain_absent(monkeypatch, tmp_path):
    """No mrg_build but Docker present => run the image, parse its JSON."""
    monkeypatch.setattr(_local_build, "_have_local_toolchain", lambda: False)
    monkeypatch.setattr(_local_build.shutil, "which", lambda name: "/usr/bin/docker")

    design = tmp_path / "design.py"
    design.write_text("# design\n")
    seen = {}

    def fake_run(cmd, capture_output, text):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(_REPORT), "")

    monkeypatch.setattr(_local_build.subprocess, "run", fake_run)

    rep = _local_build.pnr(design, target_mhz=100)
    assert isinstance(rep, BuildReport) and rep.fmax_mhz == 110.0 and rep.scope == "soc"
    assert seen["cmd"][0] == "docker" and "pnr" in seen["cmd"]
    assert "--target-mhz" in seen["cmd"] and "100" in seen["cmd"]


def test_raises_when_neither_backend_available(monkeypatch, tmp_path):
    monkeypatch.setattr(_local_build, "_have_local_toolchain", lambda: False)
    monkeypatch.setattr(_local_build.shutil, "which", lambda name: None)
    design = tmp_path / "design.py"
    design.write_text("x")
    with pytest.raises(_local_build.SandboxUnavailableError):
        _local_build.synth(design)
