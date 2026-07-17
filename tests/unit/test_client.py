"""Unit tests for _client.submit()'s multipart form construction.

No network is used -- requests.post is faked to capture what would be sent.
"""

from __future__ import annotations

from manhattan_reasoning_gym import _client


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"job_id": "job-123"}


def test_submit_omits_top_when_not_given(monkeypatch, tmp_path):
    design = tmp_path / "design.py"
    design.write_text("# design\n")
    seen = {}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(str(design), "key", "https://api.example")
    assert seen["data"] is None


def test_submit_includes_top_when_given(monkeypatch, tmp_path):
    design = tmp_path / "design.v"
    design.write_text("module m; endmodule\n")
    seen = {}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(str(design), "key", "https://api.example", top="echo_slave")
    assert seen["data"] == {"top": "echo_slave"}


def test_submit_combines_top_with_other_fields(monkeypatch, tmp_path):
    design = tmp_path / "design.v"
    design.write_text("module m; endmodule\n")
    seen = {}

    def fake_post(url, headers, files, data):
        seen["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(_client.requests, "post", fake_post)
    _client.submit(
        str(design), "key", "https://api.example",
        top="echo_slave", sys_clk_freq=90_000_000,
    )
    assert seen["data"] == {"top": "echo_slave", "sys_clk_freq": "90000000"}
