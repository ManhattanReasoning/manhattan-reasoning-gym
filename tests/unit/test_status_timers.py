import argparse
import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
import requests

from manhattan_reasoning_gym import _cli, _client

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-10-11T11:58:55+00:00", "00:01:05"),
        ("2026-10-09T11:00:00Z", "49:00:00"),
        ("2026-10-11T07:00:00-04:00", "01:00:00"),
        ("2026-10-11T12:01:00Z", "00:00:00"),
        ("2026-10-11T11:00:00", "-"),
        ("invalid", "-"),
        (None, "-"),
    ],
)
def test_elapsed_is_timezone_aware_and_handles_old_servers(value, expected):
    assert _cli._elapsed_since(value, NOW) == expected


@pytest.mark.parametrize("status", ["complete", "failed", "cancelled"])
def test_finished_job_timer_stops_at_last_update(status):
    job = {
        "status": status,
        "created_at": "2026-10-10T10:00:00Z",
        "updated_at": "2026-10-10T10:02:00Z",
    }
    assert _cli._job_elapsed(job, NOW) == "00:02:00"


@pytest.fixture
def setup_status(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(_cli, "datetime", Clock)
    monkeypatch.setattr(
        _cli, "_creds_optional", lambda args: ("key", "https://example")
    )
    board = {
        "fpga_id": 0,
        "state": "reserved",
        "owner": "alice",
        "session": {"created_at": "2026-10-11T11:30:00Z"},
        "current_job_id": None,
    }
    monkeypatch.setattr(_client, "list_fpgas", lambda *args: [board])
    monkeypatch.setattr(_client, "get_fpga", lambda *args: board)
    jobs = Mock(
        return_value=[
            {
                "job_id": "abcdefgh-full",
                "fpga_id": None,
                "type": "build_and_program",
                "status": "running",
                "created_at": "2026-10-11T11:55:00Z",
                "build_stage": "nextpnr",
            },
            {"job_id": "finished", "type": "build_and_program", "status": "complete"},
        ]
    )
    monkeypatch.setattr(_client, "list_jobs", jobs)
    return board, jobs


@pytest.mark.parametrize("fpga_id", [None, 0])
def test_status_displays_reservation_timer(setup_status, capsys, fpga_id):
    board, jobs = setup_status
    _cli.cmd_status(argparse.Namespace(fpga_id=fpga_id, json=False))
    out = capsys.readouterr().out
    assert "00:30:00" in out
    if fpga_id is None:
        assert "00:05:00" in out and "nextpnr" in out
        assert "finished" not in out
        jobs.assert_called_once_with("key", "https://example", timeout=3)
    else:
        jobs.assert_not_called()


def test_missing_legacy_timestamps_show_unknown(setup_status, capsys):
    board, _ = setup_status
    board["session"] = {"expires_at": "2026-10-11T13:00:00Z"}
    _cli.cmd_status(argparse.Namespace(fpga_id=0, json=False))
    assert "held for:      -" in capsys.readouterr().out


@pytest.mark.parametrize("fpga_id", [None, 0])
def test_json_preserves_response_without_extra_requests(setup_status, capsys, fpga_id):
    board, jobs = setup_status
    _cli.cmd_status(argparse.Namespace(fpga_id=fpga_id, json=True))
    assert json.loads(capsys.readouterr().out) == (
        [board] if fpga_id is None else board
    )
    jobs.assert_not_called()


def test_failed_jobs_lookup_does_not_hide_board_status(setup_status, capsys):
    _, jobs = setup_status
    jobs.side_effect = requests.Timeout()
    _cli.cmd_status(argparse.Namespace(fpga_id=None, json=False))
    output = capsys.readouterr().out
    assert "reserved" in output and "active builds: unavailable" in output


def test_anonymous_status_does_not_fetch_private_jobs(
    setup_status, monkeypatch, capsys
):
    _, jobs = setup_status
    monkeypatch.setattr(_cli, "_creds_optional", lambda args: (None, "https://example"))
    _cli.cmd_status(argparse.Namespace(fpga_id=None, json=False))
    jobs.assert_not_called()
    assert "reserved" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["jobs", "job"])
@pytest.mark.parametrize("as_json", [False, True])
def test_job_commands_show_elapsed_without_changing_json(
    monkeypatch, capsys, command, as_json
):
    job = {
        "job_id": "abcdefgh-full",
        "fpga_id": None,
        "type": "build_and_program",
        "status": "complete",
        "created_at": "2026-10-10T10:00:00Z",
        "updated_at": "2026-10-10T10:02:00Z",
    }
    monkeypatch.setattr(_cli, "_creds", lambda args: ("key", "https://example"))
    monkeypatch.setattr(
        _cli, "_creds_optional", lambda args: ("key", "https://example")
    )
    monkeypatch.setattr(_client, "list_jobs", lambda *args, **kwargs: [job])
    monkeypatch.setattr(_client, "get_job", lambda *args: job)
    args = argparse.Namespace(job_id=job["job_id"], status=None, json=as_json)
    getattr(_cli, f"cmd_{command}")(args)
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output) == ([job] if command == "jobs" else job)
    else:
        assert "00:02:00" in output
