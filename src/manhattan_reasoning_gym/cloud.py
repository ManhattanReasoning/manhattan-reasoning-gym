"""Direct cloud silicon — for a user or unrestricted agent holding an API key.

Talks to the orchestrator directly (needs ``MRG_API_KEY`` + network) to program
and drive a real FPGA: define an ``App`` and ``mrg run`` it, or use the session
helpers. **Sandboxed agents do not use this surface** — they have no key and no
egress; they reach silicon through the broker (see ``mrg.sandbox``).
"""

from __future__ import annotations

import os

from ._app import App, RegisterMap
from ._client import NoFPGAAvailableError, Stream, get_session, release_session

__all__ = [
    "App",
    "RegisterMap",
    "Stream",
    "get_session",
    "release_session",
    "NoFPGAAvailableError",
    "secret",
]


def secret(env_var: str) -> str:
    """Read a required environment variable (for API keys, etc.).

    Raises ``ValueError`` immediately if the variable is not set, so the error
    surfaces at import time rather than at first use.

    Example::

        app = manhattan_reasoning_gym.cloud.App(
            "my_design",
            design="design.py",
            fpga_id=0,
            api_key=manhattan_reasoning_gym.cloud.secret("MRG_API_KEY"),
        )
    """
    val = os.environ.get(env_var)
    if val is None:
        raise ValueError(
            f"Required environment variable {env_var!r} is not set.\n"
            f"  export {env_var}=<your-api-key>"
        )
    return val
