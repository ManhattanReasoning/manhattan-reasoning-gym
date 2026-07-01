"""Promote a design to silicon — the in-container side of the broker channel.

The agent, having vetted a candidate locally, calls ``promote(design, report)``.
This does **not** call the cloud — the container has no key and no network. It
drops a request file in the shared workspace and blocks until the trusted side
(the ``Sandbox`` running outside the container) writes a response.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def _workspace() -> Path:
    return Path(os.environ.get("MRG_WORKSPACE", "/work"))


def promote(
    design: str | Path,
    report: Any,
    *,
    agent: str = "default",
    workspace: str | Path | None = None,
    timeout: float = 900.0,
    poll_interval: float = 0.5,
) -> dict:
    """Request a silicon run for ``design``, blocking until the broker responds.

    Args:
        design: path to the design.py being promoted.
        report: the local BuildReport for it (object or dict), forwarded to the
            trusted side (e.g. for a custom ``guard`` or for logging).
        agent: caller id (forwarded to the trusted side; e.g. for a guard).
        workspace: shared dir (default: $MRG_WORKSPACE or /work).
        timeout: seconds to wait for the broker.

    Returns:
        The broker's response dict: ``{accepted, reason, silicon?}``.
    """
    ws = Path(workspace) if workspace else _workspace()
    promote_dir = ws / "promote"
    promote_dir.mkdir(parents=True, exist_ok=True)

    rid = uuid.uuid4().hex[:12]
    design_bytes = Path(design).read_bytes()
    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    request = {
        "id": rid,
        "agent": agent,
        "design_b64": base64.b64encode(design_bytes).decode("ascii"),
        "report": report_dict,
    }

    # Atomic publish: write to a temp name, then rename, so the broker never
    # reads a half-written request.
    req_path = promote_dir / f"{rid}.request.json"
    tmp = promote_dir / f".{rid}.request.json.tmp"
    tmp.write_text(json.dumps(request))
    os.replace(tmp, req_path)

    resp_path = promote_dir / f"{rid}.response.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if resp_path.exists():
            return json.loads(resp_path.read_text())
        time.sleep(poll_interval)
    raise TimeoutError(f"broker did not respond to promote {rid} within {timeout}s")
