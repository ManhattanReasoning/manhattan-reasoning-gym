import pytest

import manhattan_reasoning_gym
from manhattan_reasoning_gym import _client


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("MRG_API_KEY", "test-key")
    return tmp_path


# ── App board discovery ──────────────────────────────────────────────────────
#
# A fresh build never picks a board itself -- the server assigns whichever one
# frees up first once the build finishes, dispatching it inline against a
# build slot (not a board) at submit time. So there's nothing to pin or
# auto-select before building; app.fpga_id is discovered from the completed
# job, not chosen. fpga_id is still a meaningful App() argument for the
# --no-program reconnect case (skip building, talk to a board you already
# have a session on), which these tests also cover.


def test_fpga_id_is_none_until_programmed(monkeypatch):
    monkeypatch.setattr(_client, "submit", lambda *a, **kw: "job")
    monkeypatch.setattr(
        _client, "poll_job", lambda *a, **kw: {"status": "complete", "fpga_id": 2}
    )

    app = manhattan_reasoning_gym.App("x", design="d.py")
    assert app.fpga_id is None
    app._program()

    assert app.fpga_id == 2


def test_submit_is_never_given_a_board(monkeypatch):
    calls = {}

    def fake_submit(design_path, api_key, api_url, **kw):
        calls["design_path"] = design_path
        return "job"

    monkeypatch.setattr(_client, "submit", fake_submit)
    monkeypatch.setattr(
        _client, "poll_job", lambda *a, **kw: {"status": "complete", "fpga_id": 4}
    )

    app = manhattan_reasoning_gym.App("x", design="d.py")
    app._program()

    assert calls["design_path"] == "d.py"
    assert app.fpga_id == 4


def test_pinned_fpga_id_is_overwritten_once_a_build_runs(monkeypatch):
    """fpga_id passed to App() is for --no-program reconnect, not a build hint
    -- a build that does run must still overwrite it with the real board."""
    monkeypatch.setattr(_client, "submit", lambda *a, **kw: "job")
    monkeypatch.setattr(
        _client, "poll_job", lambda *a, **kw: {"status": "complete", "fpga_id": 5}
    )

    app = manhattan_reasoning_gym.App("x", design="d.py", fpga_id=3)
    app._program()

    assert app.fpga_id == 5


def test_unprogrammed_pin_survives_with_no_program():
    """The --no-program reconnect path never calls _program() at all, so a
    pinned id must be left exactly as given."""
    app = manhattan_reasoning_gym.App("x", design="d.py", fpga_id=7)
    assert app.fpga_id == 7


# ── timing-target plumbing ───────────────────────────────────────────────────

def test_app_forwards_timing_target_to_submit(monkeypatch):
    calls = {}

    def fake_submit(design_path, api_key, api_url, **kw):
        calls.update(kw)
        return "job"

    monkeypatch.setattr(_client, "submit", fake_submit)
    monkeypatch.setattr(
        _client, "poll_job", lambda *a, **kw: {"status": "complete", "fpga_id": 1}
    )

    app = manhattan_reasoning_gym.App("x", design="d.py", timing_target_mhz=90)
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

    _client.submit(str(design), "k", "u", timing_target_mhz=90)
    assert seen["data"] == {"timing_target_mhz": "90"}

    # Nothing set => no form body at all (older servers get an unchanged request).
    _client.submit(str(design), "k", "u")
    assert seen["data"] is None
