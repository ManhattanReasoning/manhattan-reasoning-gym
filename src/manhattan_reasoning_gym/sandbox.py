"""Sandboxed-agent silicon — reach a real board through the broker.

For an agent running in the locked-down sandbox (no API key, no network). Having
vetted a candidate locally with ``mrg.build``, it ``promote``s the design; the
trusted side (outside the container) runs it on silicon. This is the ONLY way a
sandboxed agent reaches hardware — it never holds a key or calls the
orchestrator directly (that's ``mrg.cloud``).
"""

from __future__ import annotations

from ._broker_client import promote

__all__ = ["promote"]
