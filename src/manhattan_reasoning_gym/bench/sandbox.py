"""Declare a sandbox, run it — the one operator-facing object.

A :class:`Sandbox` bundles everything the trusted side needs to run a sandboxed
agent: the **isolation** profile for the untrusted container, the **silicon**
backend the promote channel reaches, and (optionally) a **guard** you supply to
vet promotes. You declare it with your params and call :meth:`Sandbox.run`; the
broker that shuttles promote requests to silicon is wired up internally — there
is no separate broker to stand up.

    sb = Sandbox(files=["design.py", "agent.py"])   # mock silicon unless a key is set
    result = sb.run("agent.py")
    for p in result.promotions:
        print(p)

**No gating by default.** The framework does not decide whether a promote is
"good enough" — the agent gates itself in its own code, and if you want a
promote check you pass ``guard=``. We just provide the playground.

**Trust boundary (unchanged):** the container has no key and no network; promote
is a file handoff on the shared workspace; only this trusted process holds the
key and touches silicon.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .launcher import SandboxProfile, run_sandbox
from .silicon import CloudSilicon

# A silicon backend: (design_bytes, request) -> result dict.
SiliconFn = Callable[[bytes, dict], dict]
# An optional promote guard: (design_bytes, report) -> reject reason, or None to
# accept. Opt-in only — there is no built-in gate.
Guard = Callable[[bytes, dict], "str | None"]


def _no_silicon(design_bytes: bytes, request: dict) -> dict:
    """Default backend when no key is configured — honest no-op, no board."""
    return {
        "status": "no_silicon",
        "note": "no silicon configured; set api_key / MRG_API_KEY or pass "
        "silicon= to reach real hardware",
    }


@dataclass
class SandboxResult:
    """Outcome of a :meth:`Sandbox.run`."""

    returncode: int
    stdout: str
    stderr: str
    promotions: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class _PromoteBroker:
    """Internal plumbing: shuttle promote requests to silicon.

    Not a public concept — a :class:`Sandbox` wires one up for you. Watches
    ``<workspace>/promote/`` for ``*.request.json`` dropped by the in-container
    ``mrg.sandbox.promote`` and writes a ``*.response.json`` back for each.
    """

    def __init__(self, silicon: SiliconFn, guard: Guard | None = None) -> None:
        self.silicon = silicon
        self.guard = guard

    def poll_once(self, workspace: Path | str) -> list[dict]:
        """Handle every currently-pending request; return the responses made."""
        promote_dir = Path(workspace) / "promote"
        if not promote_dir.exists():
            return []
        responses: list[dict] = []
        for req_path in sorted(promote_dir.glob("*.request.json")):
            rid = req_path.name[: -len(".request.json")]
            resp_path = promote_dir / f"{rid}.response.json"
            if resp_path.exists():
                continue
            resp = self._handle(req_path)
            # Atomic: temp file then rename, so the agent never reads a partial.
            tmp = resp_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(resp))
            os.replace(tmp, resp_path)
            responses.append(resp)
        return responses

    def _handle(self, req_path: Path) -> dict:
        req = json.loads(req_path.read_text())
        rid = req["id"]
        design = base64.b64decode(req["design_b64"])
        if self.guard is not None:
            reason = self.guard(design, req.get("report", {}))
            if reason:
                return {"id": rid, "accepted": False, "reason": reason}
        result = self.silicon(design, req)
        return {"id": rid, "accepted": True, "reason": "ok", "silicon": result}


def _resolve_profile(
    isolation: str | SandboxProfile, image: str | None
) -> SandboxProfile:
    if isinstance(isolation, SandboxProfile):
        profile = isolation
    elif isolation == "locked":
        profile = SandboxProfile.locked()
    elif isolation == "dev":
        profile = SandboxProfile.dev()
    else:
        raise ValueError(
            f"unknown isolation {isolation!r}; use 'locked', 'dev', or a SandboxProfile"
        )
    if image:
        profile.image = image
    return profile


def _resolve_silicon(
    silicon: str | SiliconFn,
    api_key: str | None,
    api_url: str | None,
    sys_clk_freq: int | None,
) -> SiliconFn:
    if callable(silicon):
        return silicon
    key = api_key or os.environ.get("MRG_API_KEY")
    if silicon == "mock":
        return _no_silicon
    if silicon == "cloud":
        if not key:
            raise ValueError("silicon='cloud' needs api_key or MRG_API_KEY")
        return CloudSilicon(key, api_url, sys_clk_freq=sys_clk_freq)
    if silicon == "auto":
        if key:
            return CloudSilicon(key, api_url, sys_clk_freq=sys_clk_freq)
        return _no_silicon
    raise ValueError(
        f"unknown silicon {silicon!r}; use 'auto', 'cloud', 'mock', or a callable"
    )


class Sandbox:
    """A locked-down place to run a sandboxed agent, with a wired-up silicon path.

    Args:
        files: files copied into the container's ``/work`` (your ``design.py``,
            the agent entrypoint, any helpers).
        isolation: ``"locked"`` (default; untrusted, no net/key/ro-root),
            ``"dev"`` (trusted local poking), or a :class:`SandboxProfile`.
        silicon: ``"auto"`` (default: real cloud if a key is set, else a no-op),
            ``"cloud"``, ``"mock"``, or your own ``SiliconFn``.
        api_key / api_url / sys_clk_freq: passed to the cloud silicon backend.
            The key stays in this trusted process — never in the container.
        guard: optional ``(design_bytes, report) -> reject-reason | None`` to vet
            promotes. Default ``None`` = every promote goes straight through.
        image: override the sandbox docker image.
    """

    def __init__(
        self,
        files: Sequence[str | Path] = (),
        *,
        isolation: str | SandboxProfile = "locked",
        silicon: str | SiliconFn = "auto",
        api_key: str | None = None,
        api_url: str | None = None,
        sys_clk_freq: int | None = None,
        guard: Guard | None = None,
        image: str | None = None,
        poll_interval: float = 0.2,
    ) -> None:
        self.files = [Path(f) for f in files]
        self.profile = _resolve_profile(isolation, image)
        self._broker = _PromoteBroker(
            _resolve_silicon(silicon, api_key, api_url, sys_clk_freq), guard
        )
        self.poll_interval = poll_interval

    def run(self, entrypoint: str | Path, *, timeout: int = 1800) -> SandboxResult:
        """Launch the agent in the container and broker its promotes to silicon.

        Creates a throwaway workspace, copies :attr:`files` in, runs
        ``python /work/<entrypoint>`` under the isolation profile while an
        internal loop answers promotes, then tears the workspace down.
        """
        workspace = Path(tempfile.mkdtemp(prefix="mrg_sandbox_"))
        try:
            for f in self.files:
                shutil.copy(f, workspace / f.name)

            entry = Path(entrypoint).name
            promotions: list[dict] = []
            stop = threading.Event()

            def poll_loop() -> None:
                while not stop.is_set():
                    promotions.extend(self._broker.poll_once(workspace))
                    time.sleep(self.poll_interval)

            poller = threading.Thread(target=poll_loop, daemon=True)
            poller.start()
            try:
                proc = run_sandbox(
                    ["python", f"/work/{entry}"],
                    workspace=workspace,
                    profile=self.profile,
                    timeout=timeout,
                )
            finally:
                stop.set()
                poller.join(timeout=2)
            # Drain any promote that landed in the final tick.
            promotions.extend(self._broker.poll_once(workspace))

            return SandboxResult(
                proc.returncode, proc.stdout, proc.stderr, promotions
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
