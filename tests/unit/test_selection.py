import pytest

import manhattan_reasoning_gym
from manhattan_reasoning_gym import _app, _client


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("MRG_API_KEY", "test-key")
    return tmp_path


# ── find_idle_fpga ───────────────────────────────────────────────────────────

def test_find_idle_returns_lowest_idle(monkeypatch):
    monkeypatch.setattr(_client, "list_fpgas", lambda k, u: [
        {"fpga_id": 0, "state": "reserved"},
        {"fpga_id": 1, "state": "idle"},
        {"fpga_id": 2, "state": "idle"},
    ])
    assert _client.find_idle_fpga("k", "u") == 1


def test_find_idle_raises_when_none_idle(monkeypatch):
    monkeypatch.setattr(_client, "list_fpgas", lambda k, u: [
        {"fpga_id": 0, "state": "reserved"},
        {"fpga_id": 1, "state": "building"},
    ])
    with pytest.raises(_client.NoFPGAAvailableError) as exc:
        _client.find_idle_fpga("k", "u")
    # The error should be actionable: list current states.
    assert "0=reserved" in str(exc.value)


def test_find_idle_raises_on_empty_list(monkeypatch):
    monkeypatch.setattr(_client, "list_fpgas", lambda k, u: [])
    with pytest.raises(_client.NoFPGAAvailableError):
        _client.find_idle_fpga("k", "u")


def test_error_is_exported():
    assert manhattan_reasoning_gym.NoFPGAAvailableError is _client.NoFPGAAvailableError


# ── App auto-selection ───────────────────────────────────────────────────────

def test_program_auto_selects_when_unpinned(monkeypatch):
    calls = {}

    def fake_submit(fpga_id, *a, **kw):
        calls["submit_fpga_id"] = fpga_id
        return "job"

    monkeypatch.setattr(_client, "find_idle_fpga", lambda k, u: 2)
    monkeypatch.setattr(_client, "submit", fake_submit)
    monkeypatch.setattr(_client, "poll_job", lambda *a, **kw: None)

    app = manhattan_reasoning_gym.App("x", design="d.py")
    assert app.fpga_id is None
    app._program()

    assert app.fpga_id == 2
    # submit() must have been called with the resolved id, not None.
    assert calls["submit_fpga_id"] == 2


def test_program_keeps_pinned_id(monkeypatch):
    monkeypatch.setattr(
        _client, "find_idle_fpga",
        lambda k, u: (_ for _ in ()).throw(AssertionError("should not auto-select")),
    )
    monkeypatch.setattr(_client, "submit", lambda *a, **kw: "job")
    monkeypatch.setattr(_client, "poll_job", lambda *a, **kw: None)

    app = manhattan_reasoning_gym.App("x", design="d.py", fpga_id=5)
    app._program()
    assert app.fpga_id == 5


def test_resolve_is_cached(monkeypatch):
    seq = iter([3, 4])
    monkeypatch.setattr(_app._client, "find_idle_fpga", lambda k, u: next(seq))
    app = manhattan_reasoning_gym.App("x", design="d.py")
    app._resolve_fpga()
    app._resolve_fpga()  # second call must not re-pick
    assert app.fpga_id == 3


# ── timing-target plumbing ───────────────────────────────────────────────────

def test_app_forwards_timing_target_to_submit(monkeypatch):
    calls = {}

    def fake_submit(fpga_id, design, api_key, api_url, **kw):
        calls.update(kw)
        return "job"

    monkeypatch.setattr(_client, "submit", fake_submit)
    monkeypatch.setattr(_client, "poll_job", lambda *a, **kw: None)

    app = manhattan_reasoning_gym.App(
        "x", design="d.py", fpga_id=1, timing_target_mhz=90
    )
    app._program()
    assert calls["timing_target_mhz"] == 90


def test_app_timing_target_from_env(monkeypatch):
    monkeypatch.setenv("MRG_TIMING_TARGET_MHZ", "75")
    app = manhattan_reasoning_gym.App("x", design="d.py")
    assert app.timing_target_mhz == 75.0


def test_submit_sends_timing_target_form_field(monkeypatch, tmp_path):
    design = tmp_path / "d.py"
    design.write_text("# design\n")
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"job_id": "job"}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _Resp()

    monkeypatch.setattr(_client.requests, "post", fake_post)

    _client.submit(1, str(design), "k", "u", timing_target_mhz=90)
    assert seen["data"] == {"timing_target_mhz": "90"}

    # Nothing set => no form body at all (older servers get an unchanged request).
    _client.submit(1, str(design), "k", "u")
    assert seen["data"] is None
