"""Trusted-side harness for running sandboxed agents against silicon (step 4).

This package runs OUTSIDE the sandbox container. You declare a :class:`Sandbox`
with your isolation + silicon params and call ``.run()``; it launches the
untrusted container under a locked-down profile and brokers the agent's
promote-to-silicon requests internally. Nothing here runs untrusted agent code.

No promote gating is imposed by default — the agent decides what to promote, and
an operator can pass ``guard=`` if they want a check.
"""

from .launcher import DEFAULT_IMAGE, SandboxProfile, run_sandbox
from .sandbox import Guard, Sandbox, SandboxResult, SiliconFn
from .silicon import CloudSilicon

__all__ = [
    "Sandbox",
    "SandboxResult",
    "SandboxProfile",
    "run_sandbox",
    "DEFAULT_IMAGE",
    "CloudSilicon",
    "SiliconFn",
    "Guard",
]
