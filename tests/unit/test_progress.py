import io

import pytest

from manhattan_reasoning_gym import _client, _progress


def _bar(color):
    s = io.StringIO()
    b = _progress.BuildProgress("demo", 3, stream=s)
    b.color = color
    return b, s


# ── BuildProgress: non-TTY fallback ──────────────────────────────────────────

def test_non_tty_prints_each_phase_once():
    b, s = _bar(False)
    for st in ["queued", "queued", "building", "building"]:
        b.update(st)
    out = s.getvalue()
    assert out.count("queued") == 1            # no duplicate spam
    assert "building bitstream" in out
    assert "\r" not in out                     # no animation control chars

def test_programming_phase_interpolates_fpga_id():
    b, s = _bar(False)
    b.update("programming")
    assert "flashing fpga3" in s.getvalue()

def test_unknown_status_passes_through():
    b, s = _bar(False)
    b.update("weird")
    assert "weird" in s.getvalue()

def test_finish_reports_ready_with_fpga():
    b, s = _bar(False)
    b.finish()
    out = s.getvalue()
    assert "ready" in out and "fpga3" in out


# ── BuildProgress: TTY rendering ─────────────────────────────────────────────

def test_tty_renders_spinner_clock_and_ansi():
    b, s = _bar(True)
    b.update("building")
    out = s.getvalue()
    assert "\r" in out and "\033[" in out                  # redraw + color
    assert any(f in out for f in _progress._FRAMES)        # a spinner frame
    assert "building bitstream" in out

def test_abort_clears_line_on_tty():
    b, s = _bar(True)
    b.abort()
    assert "\033[K" in s.getvalue()

def test_quip_shimmer_uses_256color_and_cycles():
    b, s = _bar(True)
    out = b._shimmer("hi", elapsed=0.0)
    assert "\033[38;5;" in out                       # 256-colour escape
    # Different times pick different colours (it shimmers).
    early = b._shimmer("hi", 0.0)
    later = b._shimmer("hi", 2.0)
    assert early != later

def test_rainbow_code_wraps():
    n = len(_progress._RAINBOW)
    assert _progress._rainbow_code(0) == _progress._rainbow_code(n)


# ── poll_job + on_poll integration ───────────────────────────────────────────

class _Resp:
    def __init__(self, status=None, text=""):
        self._status = status
        self.text = text
        self.ok = True

    def raise_for_status(self):
        pass

    def json(self):
        return {"status": self._status}


def test_poll_job_invokes_on_poll_and_returns_on_complete(monkeypatch):
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: _Resp("complete"))
    seen = []
    _client.poll_job(0, "j", "k", "u", on_poll=seen.append)
    assert seen == ["complete"]

def test_poll_job_raises_with_logs_on_failure(monkeypatch):
    responses = iter([_Resp(status="failed"), _Resp(text="synth blew up")])
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: next(responses))
    with pytest.raises(RuntimeError, match="synth blew up"):
        _client.poll_job(0, "j", "k", "u")

def test_poll_job_still_works_without_callback(monkeypatch):
    monkeypatch.setattr(_client.requests, "get", lambda *a, **k: _Resp("complete"))
    # Should simply return (no exception) when on_poll is omitted.
    assert _client.poll_job(0, "j", "k", "u") is None
