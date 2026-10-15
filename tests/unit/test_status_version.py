import argparse
import json
from unittest.mock import Mock

import pytest
import requests

from manhattan_reasoning_gym import _cli, _client


@pytest.mark.parametrize("fpga_id", [None, 0])
@pytest.mark.parametrize(
    "response,expected",
    [
        ({"git_commit": "a" * 40, "git_dirty": False}, "a" * 40),
        ({"git_commit": "b" * 40, "git_dirty": True}, "modified source"),
        ({"status": "ok", "redis": True}, "unknown"),
        (requests.Timeout(), "unavailable"),
    ],
)
def test_status_version_preserves_board_display(
    monkeypatch, capsys, fpga_id, response, expected
):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: (None, "https://example"))
    board = {"fpga_id": 0, "state": "idle", "session": None, "current_job_id": None}
    monkeypatch.setattr(_client, "list_fpgas", lambda *args: [board])
    monkeypatch.setattr(_client, "get_fpga", lambda *args: board)
    mock = (
        Mock(side_effect=response)
        if isinstance(response, Exception)
        else Mock(return_value=response)
    )
    monkeypatch.setattr(_client, "get_health", mock)
    _cli.cmd_status(argparse.Namespace(fpga_id=fpga_id, json=False))
    out = capsys.readouterr().out
    assert expected in out
    assert "idle" in out


@pytest.mark.parametrize("fpga_id", [None, 0])
def test_status_json_keeps_original_shape_and_skips_health(
    monkeypatch, capsys, fpga_id
):
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: (None, "https://example"))
    board = {"fpga_id": 0, "state": "idle"}
    monkeypatch.setattr(_client, "list_fpgas", lambda *args: [board])
    monkeypatch.setattr(_client, "get_fpga", lambda *args: board)
    health = Mock(side_effect=AssertionError("JSON must not fetch health"))
    monkeypatch.setattr(_client, "get_health", health)
    _cli.cmd_status(argparse.Namespace(fpga_id=fpga_id, json=True))
    assert json.loads(capsys.readouterr().out) == (
        [board] if fpga_id is None else board
    )


def test_health_uses_public_endpoint_and_timeout(monkeypatch):
    response = Mock()
    response.json.return_value = {"git_commit": "a" * 40}
    get = Mock(return_value=response)
    monkeypatch.setattr(_client.requests, "get", get)
    assert _client.get_health("https://example/") == {"git_commit": "a" * 40}
    get.assert_called_once_with("https://example/health", timeout=3)
    response.raise_for_status.assert_called_once()
