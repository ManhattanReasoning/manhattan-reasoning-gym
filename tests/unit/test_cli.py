"""Unit tests for the CLI's synth/pnr/run --top flag: parsing and defaults.

Exercises the real argparse setup in _cli.main() by monkeypatching the
dispatched cmd_* functions to capture the parsed Namespace instead of
running a real build/submit -- no toolchain, Docker, or network involved.
"""

from __future__ import annotations

import argparse
import sys

import pytest

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
        top="echo_slave", api_url=None, api_key=None, no_program=True, transport=None,
    )
    _cli.cmd_run(args)
    assert fake_app.top == "echo_slave"


# -- --transport, and the read/write commands -------------------------------


@pytest.mark.parametrize(
    "argv, attr",
    [
        (["read", "3", "0x10"], "cmd_read"),
        (["write", "3", "0x10", "0x5"], "cmd_write"),
        (["run", "app.py"], "cmd_run"),
    ],
)
def test_transport_flag_defaults_to_none(monkeypatch, argv, attr):
    assert _run_main(monkeypatch, argv, attr).transport is None


@pytest.mark.parametrize(
    "argv, attr",
    [
        (["read", "3", "0x10", "--transport", "http"], "cmd_read"),
        (["write", "3", "0x10", "0x5", "--transport", "http"], "cmd_write"),
        (["run", "app.py", "--transport", "http"], "cmd_run"),
    ],
)
def test_transport_flag_parsed(monkeypatch, argv, attr):
    assert _run_main(monkeypatch, argv, attr).transport == "http"


def test_transport_flag_rejects_unknown_values(monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["read", "3", "0x10", "--transport", "ftp"], "cmd_read")


def test_cmd_run_applies_transport_override(monkeypatch):
    """cmd_run copies --transport onto the loaded App when given."""
    from manhattan_reasoning_gym import _app

    class _FakeApp:
        def __init__(self):
            self.top = None
            self.fpga_id = None
            self.sys_clk_freq = None
            self.timing_target_mhz = None
            self.api_url = "https://api.example"
            self.api_key = "k"
            self.transport = "websocket"
            self._entrypoint = lambda: None

        def _program(self):
            pass

    fake_app = _FakeApp()
    monkeypatch.setattr(
        _cli, "_load_user_module", lambda path: _app._registry.append(fake_app)
    )
    args = argparse.Namespace(
        file="app.py", fpga_id=None, sys_clk_freq=None, timing_target_mhz=None,
        top=None, api_url=None, api_key=None, no_program=True, transport="http",
    )
    _cli.cmd_run(args)
    assert fake_app.transport == "http"


class _FakeTransport:
    def __init__(self, read_result=None, error=None):
        self.read_result = read_result
        self.error = error
        self.calls = []
        self.closed = False

    def read(self, addr, count=1):
        self.calls.append(("read", addr, count))
        if self.error:
            raise self.error
        return self.read_result

    def write(self, addr, words, fixed_address=False):
        self.calls.append(("write", addr, words, fixed_address))
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def _patch_open_transport(monkeypatch, fake):
    opened = []

    def fake_open(kind, fpga_id, api_key, api_url):
        opened.append((kind, fpga_id, api_key, api_url))
        return fake

    monkeypatch.setattr(_cli._transport, "open_transport", fake_open)
    return opened


def _rw_args(**overrides):
    base = dict(
        fpga_id=3, address=0x10, count=1, value=0x5, transport=None,
        api_key="k", api_url="https://api.example",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_read_uses_one_transport_and_closes_it(monkeypatch, capsys):
    fake = _FakeTransport(read_result=[0x2A, 0x2B])
    opened = _patch_open_transport(monkeypatch, fake)
    _cli.cmd_read(_rw_args(count=2, transport="http"))
    assert opened == [("http", 3, "k", "https://api.example")]
    assert fake.calls == [("read", 0x10, 2)]
    assert fake.closed
    assert capsys.readouterr().out.splitlines() == [
        "  0x10: 0x0000002a",
        "  0x14: 0x0000002b",
    ]


def test_cmd_write_uses_one_transport_and_closes_it(monkeypatch):
    fake = _FakeTransport()
    opened = _patch_open_transport(monkeypatch, fake)
    _cli.cmd_write(_rw_args())
    assert opened == [(None, 3, "k", "https://api.example")]
    assert fake.calls == [("write", 0x10, [0x5], False)]
    assert fake.closed


@pytest.mark.parametrize("cmd", ["cmd_read", "cmd_write"])
def test_connection_errors_exit_with_a_one_line_message(monkeypatch, cmd):
    error = ConnectionError("FPGA 3 is in use by another connection")
    fake = _FakeTransport(error=error)
    _patch_open_transport(monkeypatch, fake)
    with pytest.raises(SystemExit) as info:
        getattr(_cli, cmd)(_rw_args())
    assert str(info.value) == "error: FPGA 3 is in use by another connection"
    assert fake.closed


def test_bad_mrg_transport_env_exits_with_a_message(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", "ftp")
    with pytest.raises(SystemExit) as info:
        _cli.cmd_read(_rw_args())
    assert "unknown transport 'ftp'" in str(info.value)
