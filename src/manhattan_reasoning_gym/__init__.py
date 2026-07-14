"""Manhattan Reasoning Gym SDK.

The surface is organized by role into four namespaces:

- ``mrg.build``   — local synth/PnR feedback (in the sandbox image; no cloud).
- ``mrg.sandbox`` — brokered silicon for a sandboxed agent (``promote``; no key).
- ``mrg.cloud``   — direct cloud silicon for a key-holding user/agent (App, run).
- ``mrg.bench``   — the RL-gym harness: declare a ``Sandbox`` and run a
  sandboxed agent; promotes are brokered to silicon internally (Sandbox,
  run_sandbox, SandboxProfile). No promote gating unless you pass ``guard=``.

Rule of thumb: a **cloud user / unrestricted agent** uses ``mrg.build`` +
``mrg.cloud``; a **sandboxed agent** uses ``mrg.build`` + ``mrg.sandbox`` and
never ``mrg.cloud``; a **gym operator** running the benchmark uses ``mrg.bench``.

**Least privilege in the sandbox image:** ``cloud`` and ``bench`` are the
operator/user surfaces. They are *stripped* from the untrusted sandbox image
(see ``sandbox/Dockerfile``), so agent code there can import only ``build`` +
``sandbox``. They're imported optionally below so the package still loads when
absent.

The old flat names (``mrg.App``, ``mrg.synth``, ``mrg.promote``, ``mrg.secret``,
…) remain as back-compat aliases, but the namespaced form is canonical.
"""

from __future__ import annotations

# Always present — the sandboxed-agent surface.
from . import build, sandbox
from .build import SandboxUnavailableError, pnr, synth
from .sandbox import promote

__all__ = [
    "build",
    "sandbox",
    "synth",
    "pnr",
    "promote",
    "SandboxUnavailableError",
]

# Operator/user surfaces — stripped from the sandbox image. Import optionally so
# the package still loads with only build + sandbox present.
try:
    from . import cloud
    from .cloud import App, RegisterMap, get_session, release_session, secret

    __all__ += [
        "cloud", "App", "RegisterMap",
        "get_session", "release_session", "secret",
    ]
except ImportError:  # cloud stripped (sandbox image)
    pass

try:
    from . import bench
    from .bench import Sandbox

    __all__ += ["bench", "Sandbox"]
except ImportError:  # bench stripped (sandbox image)
    pass
