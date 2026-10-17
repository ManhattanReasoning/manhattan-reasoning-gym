"""`mrg jobs`: most recent builds by default, with type and count controls."""

import argparse
import json

import pytest

from manhattan_reasoning_gym import _cli, _client


def _job(i: int, job_type: str = "build_and_program") -> dict:
    return {
        "job_id": f"{i:08d}-full",
        "fpga_id": None,
        "type": job_type,
        "status": "complete",
        "created_at": "2026-10-10T10:00:00Z",
        "updated_at": "2026-10-10T10:02:00Z",
    }


# Newest first, as the server returns them: 15 builds interleaved with runs.
JOBS = [
    _job(i, "build_and_program" if i % 3 == 0 else "run") for i in range(45)
]
BUILDS = [j for j in JOBS if j["type"] == "build_and_program"]


@pytest.fixture
def run_jobs(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "_creds", lambda args: ("key", "https://example"))
    monkeypatch.setattr(_client, "list_jobs", lambda *a, **k: JOBS)

    def run(*argv: str) -> str:
        args = _cli_args("jobs", *argv)
        _cli.cmd_jobs(args)
        return capsys.readouterr().out

    return run


def _cli_args(*argv: str) -> argparse.Namespace:
    """Parse argv with the real parser, so defaults match `mrg jobs`."""
    import sys

    old = sys.argv
    captured = {}
    try:
        sys.argv = ["mrg", *argv]
        # main() parses then dispatches; capture the parsed args instead.
        real = _cli.cmd_jobs
        _cli.cmd_jobs = lambda a: captured.setdefault("args", a)
        try:
            _cli.main()
        finally:
            _cli.cmd_jobs = real
    finally:
        sys.argv = old
    return captured["args"]


def _ids(output: str) -> list[str]:
    return [
        line.split()[0]
        for line in output.splitlines()
        if line.strip()[:1].isdigit()
    ]


def test_default_shows_ten_most_recent_builds(run_jobs):
    out = run_jobs()
    assert _ids(out) == [_cli._fmt_id(j["job_id"]) for j in BUILDS[:10]]
    assert "run " not in out
    assert "5 older" in out and "30 of other types" in out


def test_limit_and_all(run_jobs):
    assert len(_ids(run_jobs("-n", "3"))) == 3
    out = run_jobs("--all")
    assert len(_ids(out)) == len(BUILDS)
    assert "older" not in out


def test_type_all_includes_runs(run_jobs):
    out = run_jobs("--type", "all", "--all")
    assert len(_ids(out)) == len(JOBS)
    assert "hidden" not in out


def test_no_matching_jobs_suggests_type_all(run_jobs):
    assert "no reset jobs (try --type all)" in run_jobs("--type", "reset")


def test_limit_and_all_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _cli_args("jobs", "-n", "3", "--all")


def test_json_stays_raw_and_unfiltered(run_jobs):
    assert json.loads(run_jobs("--json", "-n", "1")) == JOBS
