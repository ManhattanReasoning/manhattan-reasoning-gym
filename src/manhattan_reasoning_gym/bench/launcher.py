"""Launch the sandbox image under a chosen isolation profile.

Trusted-side code: it decides how the container is constrained. There are two
profiles, distinguished by **trust**, not by "strict vs fun":

- ``SandboxProfile.locked()`` (the default) — for **untrusted** agent code and
  benchmark eval: no network, no credentials, read-only root, dropped caps.
  Reproducible and benchmark-valid.
- ``SandboxProfile.dev()`` — for **trusted** experimentation only: network on,
  writable root, your own key forwarded. Convenient, but NOT reproducible and
  NOT benchmark-valid — never run untrusted code here.

The default (bare ``SandboxProfile()``) is the locked profile, so the safe
posture is the one you get without asking.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IMAGE = "ghcr.io/manhattanreasoning/mrg-sandbox:latest"


@dataclass
class SandboxProfile:
    """The docker run constraints applied to the container.

    Defaults are the locked-down profile: no egress, dropped capabilities,
    read-only root, bounded memory/cpu/pids. Only the workspace mount and tmpfs
    are writable. Use ``.locked()`` / ``.dev()`` for the named presets.
    """

    image: str = DEFAULT_IMAGE
    network: str = "none"  # no egress for untrusted builds
    memory: str = "8g"  # nextpnr is the memory hog
    cpus: str = "4"
    pids_limit: int = 512
    read_only_root: bool = True
    user: str | None = None  # e.g. "1000:1000"; None = image default
    # Writable scratch on the read-only rootfs. mrg_build builds in /tmp by
    # default, and HOME is pointed here so tool caches don't hit the ro root.
    tmpfs: tuple[str, ...] = ("/tmp:size=4g",)
    home: str = "/tmp"
    # Extra env vars passed into the container. EMPTY in the locked profile
    # (never inject a key into untrusted code); dev() forwards MRG_API_KEY here.
    env: dict[str, str] = field(default_factory=dict)

    def argv(
        self,
        *,
        command: list[str],
        workspace: Path | str | None = None,
        mounts: tuple[tuple[str, str, str], ...] = (),
    ) -> list[str]:
        """Build the full ``docker run`` argv for one invocation.

        ``workspace`` (if given) is mounted read-write at /work. ``mounts`` are
        extra (src, dst, mode) bind mounts.
        """
        argv = [
            "docker", "run", "--rm",
            f"--network={self.network}",
            f"--memory={self.memory}", f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}", f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
            "-e", f"HOME={self.home}",
        ]
        for key, value in self.env.items():
            argv += ["-e", f"{key}={value}"]
        if self.read_only_root:
            argv.append("--read-only")
        for t in self.tmpfs:
            argv += ["--tmpfs", t]
        if self.user:
            argv += ["--user", self.user]
        if workspace is not None:
            argv += ["-v", f"{Path(workspace).resolve()}:/work"]
        for src, dst, mode in mounts:
            argv += ["-v", f"{Path(src).resolve()}:{dst}:{mode}"]
        argv += [self.image, *command]
        return argv

    @classmethod
    def locked(cls, **overrides) -> SandboxProfile:
        """The default profile — for UNTRUSTED agent code and benchmark eval.

        No network, no credentials, read-only root, dropped caps. Reproducible
        and benchmark-valid. ``overrides`` tweak individual fields if needed.
        """
        return cls(**overrides)

    @classmethod
    def dev(cls, *, forward_api_key: bool = True, **overrides) -> SandboxProfile:
        """TRUSTED experimentation only — relaxed, WITH internet.

        For a developer (or your own agent) poking around on your machine with
        your own key: network on, writable root, and your host ``MRG_API_KEY``
        forwarded in so ``mrg.cloud`` works directly.

        Do NOT run untrusted agent code in this profile, and do NOT treat its
        results as benchmark scores — internet access makes them non-reproducible
        by definition.
        """
        env = dict(overrides.pop("env", {}))
        if forward_api_key:
            key = os.environ.get("MRG_API_KEY")
            if key:
                env["MRG_API_KEY"] = key
        overrides.setdefault("network", "bridge")
        overrides.setdefault("read_only_root", False)
        return cls(env=env, **overrides)


def run_sandbox(
    command: list[str],
    *,
    workspace: Path | str | None = None,
    profile: SandboxProfile | None = None,
    mounts: tuple[tuple[str, str, str], ...] = (),
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in the sandbox container (locked profile by default)."""
    profile = profile or SandboxProfile.locked()
    argv = profile.argv(command=command, workspace=workspace, mounts=mounts)
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
