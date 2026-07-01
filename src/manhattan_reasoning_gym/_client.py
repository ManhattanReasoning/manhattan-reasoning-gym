"""HTTP client for the Cloud FPGA orchestrator API."""

from __future__ import annotations

import time

import requests

DEFAULT_API_URL = "https://api.manhattanreasoning.com"
_RUN_POLL_INTERVAL = 0.5
_BUILD_POLL_INTERVAL = 2.0
_ANIM_INTERVAL = 0.08  # spinner tick when a progress callback is attached


class NoFPGAAvailableError(RuntimeError):
    """Raised when no idle FPGA can be found to run on."""


def find_idle_fpga(api_key: str, api_url: str) -> int:
    """Return the id of an available (idle) FPGA.

    Picks the lowest-numbered FPGA currently in the ``idle`` state. Raises
    ``NoFPGAAvailableError`` if every FPGA is busy or otherwise unavailable.

    Only as accurate as ``GET /fpga``: the orchestrator must expose just the
    physically connected boards (set ``FPGA_IDS`` server-side), otherwise this
    may select a phantom idle slot.
    """
    fpgas = list_fpgas(api_key, api_url)
    idle = sorted(f["fpga_id"] for f in fpgas if f.get("state") == "idle")
    if not idle:
        current = ", ".join(
            f"{f['fpga_id']}={f.get('state')}"
            for f in sorted(fpgas, key=lambda f: f["fpga_id"])
        ) or "no FPGAs reported"
        raise NoFPGAAvailableError(
            f"No idle FPGA available right now (current states: {current}).\n"
            "  Wait for one to free up, or pin a board with "
            "fpga_id=<n> / --fpga-id <n>."
        )
    return idle[0]


def exchange_github_token(github_token: str, api_url: str) -> dict:
    """Exchange a GitHub token for an API key.

    Returns the decoded {"api_key", "github_username"} response.
    """
    resp = requests.post(
        f"{api_url}/auth/github/exchange",
        json={"github_token": github_token},
    )
    resp.raise_for_status()
    return resp.json()


def revoke_key(api_key: str, api_url: str) -> None:
    """Revoke the calling user's API key (DELETE /auth/keys/me)."""
    resp = requests.delete(
        f"{api_url}/auth/keys/me",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()


def submit(
    fpga_id: int,
    design_path: str,
    api_key: str,
    api_url: str,
    sys_clk_freq: int | None = None,
) -> str:
    url = f"{api_url}/fpga/{fpga_id}/submit"
    # Optional multipart form field; older servers simply ignore the extra field.
    data = {"sys_clk_freq": str(sys_clk_freq)} if sys_clk_freq else None
    with open(design_path, "rb") as f:
        resp = requests.post(
            url, headers={"X-API-Key": api_key}, files={"file": f}, data=data
        )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_job(
    fpga_id: int,
    job_id: str,
    api_key: str,
    api_url: str,
    timeout: float = 2400.0,
    on_poll=None,
) -> None:
    """Block until a job completes.

    ``timeout`` is the wall-clock deadline in seconds. It defaults to 40 min so
    the client outlasts the orchestrator's gateware build ceiling
    (``BUILD_TIMEOUT_SECONDS``, 30 min by default) and surfaces the real
    ``complete``/``failed`` status instead of giving up mid-build.

    If ``on_poll`` is given it's called as ``on_poll(status)`` on a fast tick
    (~12/s) so callers can animate a spinner, while the job itself is only
    queried every ``_BUILD_POLL_INTERVAL`` seconds.
    """
    url = f"{api_url}/fpga/{fpga_id}/jobs/{job_id}"
    headers = {"X-API-Key": api_key}
    tick = _ANIM_INTERVAL if on_poll else _BUILD_POLL_INTERVAL
    deadline = time.monotonic() + timeout
    next_check = 0.0
    status = "queued"
    while time.monotonic() < deadline:
        if time.monotonic() >= next_check:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            status = resp.json()["status"]
            next_check = time.monotonic() + _BUILD_POLL_INTERVAL
            if status == "complete":
                if on_poll:
                    on_poll(status)
                return
            if status in ("failed", "cancelled"):
                logs_resp = requests.get(f"{url}/logs", headers=headers)
                logs = logs_resp.text if logs_resp.ok else "no logs available"
                raise RuntimeError(
                    f"Job {job_id!r} ended with status {status!r}.\n{logs}"
                )
        if on_poll:
            on_poll(status)
        time.sleep(tick)
    raise TimeoutError(f"Job {job_id!r} did not complete within {timeout}s")


def read(
    fpga_id: int, api_key: str, address: int, count: int, api_url: str
) -> list[int]:
    url = f"{api_url}/fpga/{fpga_id}/run"
    resp = requests.post(
        url,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"op": 2, "address": address, "data": [], "count": count},
    )
    resp.raise_for_status()
    return _poll_run(fpga_id, resp.json()["job_id"], api_key, api_url)


def write(
    fpga_id: int, api_key: str, address: int, words: list[int], api_url: str
) -> None:
    url = f"{api_url}/fpga/{fpga_id}/run"
    resp = requests.post(
        url,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"op": 1, "address": address, "data": words, "count": 0},
    )
    resp.raise_for_status()
    _poll_run(fpga_id, resp.json()["job_id"], api_key, api_url)


def list_fpgas(api_key: str, api_url: str) -> list[dict]:
    resp = requests.get(f"{api_url}/fpga", headers={"X-API-Key": api_key})
    resp.raise_for_status()
    return resp.json()


def get_fpga(fpga_id: int, api_key: str, api_url: str) -> dict:
    resp = requests.get(f"{api_url}/fpga/{fpga_id}", headers={"X-API-Key": api_key})
    resp.raise_for_status()
    return resp.json()


def get_job(fpga_id: int, job_id: str, api_key: str, api_url: str) -> dict:
    resp = requests.get(
        f"{api_url}/fpga/{fpga_id}/jobs/{job_id}",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.json()


def get_logs(fpga_id: int, job_id: str, api_key: str, api_url: str) -> str:
    resp = requests.get(
        f"{api_url}/fpga/{fpga_id}/jobs/{job_id}/logs",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.text


def cancel_job(fpga_id: int, job_id: str, api_key: str, api_url: str) -> None:
    resp = requests.delete(
        f"{api_url}/fpga/{fpga_id}/jobs/{job_id}",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()


def get_session(fpga_id: int, api_key: str, api_url: str) -> dict:
    resp = requests.get(
        f"{api_url}/fpga/{fpga_id}/session",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.json()


def release_session(fpga_id: int, api_key: str, api_url: str) -> str:
    resp = requests.post(
        f"{api_url}/fpga/{fpga_id}/session/release",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def reset_fpga(fpga_id: int, api_key: str, api_url: str) -> dict:
    resp = requests.post(
        f"{api_url}/fpga/{fpga_id}/reset",
        headers={"X-API-Key": api_key},
    )
    resp.raise_for_status()
    return resp.json()


def _poll_run(fpga_id: int, job_id: str, api_key: str, api_url: str) -> list[int]:
    status_url = f"{api_url}/fpga/{fpga_id}/jobs/{job_id}"
    result_url = f"{status_url}/result"
    headers = {"X-API-Key": api_key}
    for _ in range(120):
        resp = requests.get(status_url, headers=headers)
        resp.raise_for_status()
        status = resp.json()["status"]
        if status == "complete":
            resp = requests.get(result_url, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            if result.get("ok"):
                return result.get("data", [])
            raise RuntimeError(f"Run op failed: {result}")
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Run job {job_id!r} ended with status {status!r}")
        time.sleep(_RUN_POLL_INTERVAL)
    raise TimeoutError(f"Run job {job_id!r} did not complete in time")
