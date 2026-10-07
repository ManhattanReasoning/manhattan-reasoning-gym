"""How App.read/App.write and ``mrg read``/``mrg write`` reach the board.

Two transports share one small interface -- ``read``, ``write``, ``close``:

- ``WebSocketTransport`` (the default): one WebSocket to the orchestrator's
  ``/fpga/{id}/stream`` relay, opened on the first call, reused by every call
  after it, and closed once it has sat idle for ``_WS_IDLE_TIMEOUT`` seconds.
- ``HttpTransport``: every call is its own job on the orchestrator's ``/run``
  queue (``_client.read``/``_client.write``). Much slower -- the SDK polls each
  job every 0.5 s -- but plain HTTPS, for networks that block WebSockets.

Users pick HTTP with ``App(..., transport="http")``, ``$MRG_TRANSPORT``, or
``--transport http`` on the CLI. There is no automatic fallback: a WebSocket
that can't open raises, and the message says how to switch.
"""

from __future__ import annotations

import os

from . import _client

WEBSOCKET = "websocket"
HTTP = "http"
TRANSPORTS = (WEBSOCKET, HTTP)


def resolve_transport(value: str | None = None) -> str:
    """Return the transport to use: *value* > ``$MRG_TRANSPORT`` > websocket.

    Case and surrounding whitespace are ignored; anything other than
    ``"websocket"`` or ``"http"`` raises ``ValueError``.
    """
    raw = value or os.environ.get("MRG_TRANSPORT") or WEBSOCKET
    chosen = raw.strip().lower()
    if chosen not in TRANSPORTS:
        raise ValueError(
            f"unknown transport {raw!r}: expected 'websocket' or 'http'"
        )
    return chosen


class HttpTransport:
    """Each call is its own job on the orchestrator's ``/run`` queue."""

    def __init__(self, fpga_id: int, api_key: str, api_url: str) -> None:
        self._fpga_id = fpga_id
        self._api_key = api_key
        self._api_url = api_url

    def read(self, addr: int, count: int = 1) -> list[int]:
        return _client.read(self._fpga_id, self._api_key, addr, count, self._api_url)

    def write(
        self, addr: int, words: list[int], fixed_address: bool = False
    ) -> None:
        _client.write(
            self._fpga_id, self._api_key, addr, words, self._api_url,
            fixed_address=fixed_address,
        )

    def close(self) -> None:
        """Nothing to close: every call made and finished its own request."""
