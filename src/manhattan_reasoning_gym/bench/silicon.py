"""Real-silicon backend for the broker: build + program a design on a board.

A ``SiliconFn`` the trusted side uses when a key is configured (otherwise the
Sandbox falls back to a no-op). It holds the API key (never exposed to the
untrusted container) and drives the
existing SDK cloud path: submit (server claims a build slot and builds) ->
poll until a board is assigned and programmed -> release. The design bytes
come from the promote request.

Failure modes are returned as structured results (never raised) so one bad
promote doesn't crash the broker loop:
    programmed | no_board | submit_failed | build_failed | timeout

Functional correctness (running the design's testbench on the board) is
deliberately NOT here yet — that needs the task spec + golden (later). This
increment proves "the promoted design really builds and programs on real
silicon."
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import requests


class CloudSilicon:
    """Runs a promoted design on a real cloud FPGA. Callable as a SiliconFn."""

    def __init__(
        self,
        api_key: str,
        api_url: str | None = None,
        *,
        sys_clk_freq: int | None = None,
        release_after: bool = True,
        poll_timeout: float = 2400.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.sys_clk_freq = sys_clk_freq
        self.release_after = release_after
        self.poll_timeout = poll_timeout

    def __call__(self, design_bytes: bytes, request: dict) -> dict:
        # Lazy import so the broker package stays importable without the SDK
        # (e.g. for the mock-only unit tests).
        from manhattan_reasoning_gym import _client

        api_url = self.api_url or _client.DEFAULT_API_URL

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "design.py"
            path.write_bytes(design_bytes)
            try:
                job_id = _client.submit(
                    str(path), self.api_key, api_url,
                    sys_clk_freq=self.sys_clk_freq,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    return {"status": "no_board", "note": "no build capacity available"}
                return {"status": "submit_failed", "error": str(exc)[:2000]}
            except Exception as exc:
                return {"status": "submit_failed", "error": str(exc)[:2000]}

            try:
                job = _client.poll_job(
                    job_id, self.api_key, api_url, timeout=self.poll_timeout
                )
            except RuntimeError as exc:  # build/program failed (carries logs)
                return {"status": "build_failed", "job_id": job_id,
                        "error": str(exc)[:4000]}
            except TimeoutError as exc:
                return {"status": "timeout", "job_id": job_id, "error": str(exc)}

            fpga_id = job["fpga_id"]
            result = {"status": "programmed", "fpga_id": fpga_id, "job_id": job_id}

        if self.release_after:
            # Free the board even if release hiccups; the result still stands.
            try:
                _client.release_session(fpga_id, self.api_key, api_url)
                result["released"] = True
            except Exception as exc:
                result["released"] = False
                result["release_error"] = str(exc)[:500]
        return result
