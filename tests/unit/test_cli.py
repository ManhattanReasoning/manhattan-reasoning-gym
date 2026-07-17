"""Unit tests for the CLI's synth/pnr/run --top flag: parsing and defaults.

Exercises the real argparse setup in _cli.main() by monkeypatching the
dispatched cmd_* functions to capture the parsed Namespace instead of
running a real build/submit -- no toolchain, Docker, or network involved.
"""

from __future__ import annotations

import argparse
import sys

from manhattan_reasoning_gym import _cli


def _run_main(monkeypatch, argv, captured_attr):
    """Run _cli.main() with argv, capturing the dispatched cmd_*'s args."""
    captured = {}

    def fake_cmd(args):
        captured["args"] = args

    monkeypatch.setattr(_cli, captured_attr, fake_cmd)
    monkeypatch.setattr(sys, "argv", ["mrg", *argv])
    _cli.main()
    return captured["args"]


def test_synth_top_defaults_to_none(monkeypatch):
    args = _run_main(monkeypatch, ["synth", "design.py"], "cmd_synth")
    assert args.design == "design.py"
    assert args.top is None


def test_synth_top_parsed(monkeypatch):
    args = _run_main(
        monkeypatch, ["synth", "design.v", "--top", "echo_slave"], "cmd_synth"
    )
    assert args.design == "design.v"
    assert args.top == "echo_slave"


def test_pnr_top_defaults_to_none(monkeypatch):
    args = _run_main(monkeypatch, ["pnr", "design.py"], "cmd_pnr")
    assert args.top is None


def test_pnr_top_parsed(monkeypatch):
    args = _run_main(
        monkeypatch, ["pnr", "design.v", "--top", "echo_slave"], "cmd_pnr"
    )
    assert args.top == "echo_slave"


def test_run_top_override_defaults_to_none(monkeypatch):
    args = _run_main(monkeypatch, ["run", "app.py"], "cmd_run")
    assert args.top is None


def test_run_top_override_parsed(monkeypatch):
    args = _run_main(
        monkeypatch, ["run", "app.py", "--top", "echo_slave"], "cmd_run"
    )
    assert args.top == "echo_slave"


def test_cmd_run_applies_top_override(monkeypatch):
    """cmd_run itself copies args.top onto the loaded App when given."""
    from manhattan_reasoning_gym import _app

    class _FakeApp:
        def __init__(self):
            self.top = None
            self.fpga_id = None
            self.sys_clk_freq = None
            self.timing_target_mhz = None
            self.api_url = "https://api.example"
            self.api_key = "k"
            self._entrypoint = lambda: None

        def _program(self):
            pass

    fake_app = _FakeApp()
    # cmd_run clears the registry before loading the file, so the fake app
    # must be appended *by* the (faked) load, not pre-seeded.
    monkeypatch.setattr(
        _cli, "_load_user_module", lambda path: _app._registry.append(fake_app)
    )

    args = argparse.Namespace(
        file="app.py", fpga_id=None, sys_clk_freq=None, timing_target_mhz=None,
        top="echo_slave", api_url=None, api_key=None, no_program=True,
    )
    _cli.cmd_run(args)
    assert fake_app.top == "echo_slave"
