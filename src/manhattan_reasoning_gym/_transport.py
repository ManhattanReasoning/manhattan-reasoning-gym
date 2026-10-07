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
import threading
import time

import websockets.sync.client

from . import _client, _wire

WEBSOCKET = "websocket"
HTTP = "http"
TRANSPORTS = (WEBSOCKET, HTTP)

# How long to wait for the WebSocket handshake. A network that blocks
# WebSockets often stalls the upgrade rather than refusing it outright.
_WS_OPEN_TIMEOUT = 10.0


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


def open_transport(
    kind: str | None, fpga_id: int, api_key: str, api_url: str
) -> Transport:
    """Return a transport of *kind* (resolved via ``resolve_transport``).

    Never touches the network: a WebSocket opens on its first call.
    """
    if resolve_transport(kind) == HTTP:
        return HttpTransport(fpga_id, api_key, api_url)
    return WebSocketTransport(fpga_id, api_key, api_url)


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


class WebSocketTransport:
    """One WebSocket to the FPGA, opened on first use and reused after.

    Every call holds ``self._lock`` for its whole send/receive, so two threads
    sharing one App can't interleave frames on the wire -- the relay matches
    replies to requests purely by order.
    """

    def __init__(self, fpga_id: int, api_key: str, api_url: str) -> None:
        self._fpga_id = fpga_id
        self._api_key = api_key
        self._api_url = api_url
        self._lock = threading.Lock()
        self._conn: websockets.sync.client.ClientConnection | None = None
        self._last_used = 0.0

    def read(self, addr: int, count: int = 1) -> list[int]:
        """Read ``count`` 32-bit words starting at byte address ``addr``."""
        what = f"read of {count} words at {addr:#x}"
        request = _wire.WishboneRequest(
            op=_wire.WishboneOp.READ, address=addr, count=count
        )
        response = self._transact(request, what)
        if not response.ok:
            raise RuntimeError(f"{what} failed")
        return response.data

    def write(
        self, addr: int, words: list[int], fixed_address: bool = False
    ) -> None:
        """Write ``words`` from ``addr`` up (or all at ``addr`` if fixed)."""
        what = f"write to {addr:#x}"
        request = _wire.WishboneRequest(
            op=_wire.WishboneOp.WRITE,
            address=addr,
            data=words,
            fixed_address=fixed_address,
        )
        response = self._transact(request, what)
        if not response.ok:
            raise RuntimeError(f"{what} failed")

    def close(self) -> None:
        """Close the socket if it's open. Safe to call more than once."""
        with self._lock:
            self._close_locked()

    # -- internals; every *_locked method expects self._lock to be held -------

    def _transact(
        self, request: _wire.WishboneRequest, what: str
    ) -> _wire.WishboneResponse:
        payload = request.to_bytes()
        with self._lock:
            conn = self._open_locked()
            conn.send(payload)
            raw = conn.recv()
            self._last_used = time.monotonic()
        if isinstance(raw, str):
            raise RuntimeError("received an unexpected text frame")
        return _wire.WishboneResponse.from_bytes(raw)

    def _open_locked(self) -> websockets.sync.client.ClientConnection:
        if self._conn is None:
            self._conn = _connect(self._fpga_id, self._api_key, self._api_url)
            self._last_used = time.monotonic()
        return self._conn

    def _close_locked(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def _connect(
    fpga_id: int, api_key: str, api_url: str
) -> websockets.sync.client.ClientConnection:
    conn = websockets.sync.client.connect(
        _client._ws_url(api_url, fpga_id),
        additional_headers={"X-API-Key": api_key},
        open_timeout=_WS_OPEN_TIMEOUT,
    )
    # websockets >= 16 warns when a connection is used outside
    # `with connect(...)`. This one deliberately outlives any single block --
    # _close_locked closes it -- so enter its context explicitly.
    return conn.__enter__()


# What open_transport returns; App and the CLI only use read/write/close.
Transport = HttpTransport | WebSocketTransport
