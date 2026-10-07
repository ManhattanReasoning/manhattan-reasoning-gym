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

import websockets.exceptions
import websockets.sync.client

from . import _client, _wire

WEBSOCKET = "websocket"
HTTP = "http"
TRANSPORTS = (WEBSOCKET, HTTP)

# Close an idle socket after this many seconds without a call. An open stream
# holds the board's link lock on the orchestrator (state/store.py
# fpga_link_lock, timeout=30), and the server renews that lock only when an
# operation goes through -- keep-alive pings don't renew it. A socket left idle
# past 30 s outlives its lock: the next op is forwarded to the FPGA unlocked,
# then the server drops the socket. Closing it ourselves first avoids that; the
# 5 s gap covers reply latency and the close reaching the server.
#
# NOTE: if fpga_link_lock's timeout changes on the server, keep this below it.
_WS_IDLE_TIMEOUT = 25.0

# How long to wait for the WebSocket handshake. A network that blocks
# WebSockets often stalls the upgrade rather than refusing it outright.
_WS_OPEN_TIMEOUT = 10.0

# Close code the orchestrator's /stream uses after accepting a socket it then
# can't serve because another connection holds the board's link lock. (Its
# refusals before accepting -- board not reserved, not your session -- reach
# the client as an HTTP status on the handshake instead.)
_CLOSE_BUSY = 4409

_HTTP_HINT = (
    "If your network blocks WebSockets, use HTTP instead: "
    'App(..., transport="http") or MRG_TRANSPORT=http.'
)


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
    """One WebSocket to the FPGA: opened on first use, closed when idle.

    Every call holds ``self._lock`` for its whole send/receive, so the idle
    watcher can never close the socket mid-call and two threads sharing one
    App can't interleave frames on the wire -- the relay matches replies to
    requests purely by order.
    """

    def __init__(self, fpga_id: int, api_key: str, api_url: str) -> None:
        self._fpga_id = fpga_id
        self._api_key = api_key
        self._api_url = api_url
        self._idle_timeout = _WS_IDLE_TIMEOUT
        self._lock = threading.Lock()
        self._conn: websockets.sync.client.ClientConnection | None = None
        self._last_used = 0.0
        # One watcher thread per open socket; _stop wakes it early on close.
        self._stop: threading.Event | None = None
        self._watcher: threading.Thread | None = None

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
        """Close the socket and stop its idle watcher. Safe to call twice."""
        with self._lock:
            watcher = self._watcher
            self._close_locked()
        # Join outside the lock: the watcher may be waiting to take it.
        if watcher is not None:
            watcher.join(timeout=1)

    # -- internals; every *_locked method expects self._lock to be held -------

    def _transact(
        self, request: _wire.WishboneRequest, what: str
    ) -> _wire.WishboneResponse:
        payload = request.to_bytes()
        with self._lock:
            try:
                conn = self._send_locked(payload)
                raw = conn.recv()
            except websockets.exceptions.ConnectionClosed as exc:
                self._close_locked()
                raise _closed_error(exc, self._fpga_id, what) from exc
            except BaseException:
                # Interrupted mid-exchange (e.g. Ctrl-C while waiting for the
                # reply): the server may still answer, and the relay pairs
                # replies with requests purely by order, so this socket would
                # hand the next call a stale reply. Drop it.
                self._close_locked()
                raise
            self._last_used = time.monotonic()
        if isinstance(raw, str):
            raise RuntimeError("received an unexpected text frame")
        return _wire.WishboneResponse.from_bytes(raw)

    def _send_locked(self, payload: bytes) -> websockets.sync.client.ClientConnection:
        """Send *payload*, replacing a socket that closed while idle.

        Returns the connection the payload went out on.
        """
        conn = self._open_locked()
        try:
            conn.send(payload)
        except websockets.exceptions.ConnectionClosed:
            # The socket closed while idle (server restart, network drop).
            # websockets refuses to send on a socket that isn't open, so
            # nothing reached the server and a fresh socket can safely carry
            # this call. Only once: a second failure is a real one.
            self._close_locked()
            conn = self._open_locked()
            conn.send(payload)
        return conn

    def _open_locked(self) -> websockets.sync.client.ClientConnection:
        if self._conn is None:
            self._conn = _connect(self._fpga_id, self._api_key, self._api_url)
            self._last_used = time.monotonic()
            self._stop = threading.Event()
            # Daemon: a script that ends with the socket still open must not
            # hang waiting for the idle timeout.
            self._watcher = threading.Thread(
                target=self._watch_idle,
                args=(self._conn, self._stop),
                name=f"mrg-ws-idle-fpga{self._fpga_id}",
                daemon=True,
            )
            self._watcher.start()
        return self._conn

    def _close_locked(self) -> None:
        if self._stop is not None:
            self._stop.set()
        self._stop = None
        self._watcher = None
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def _watch_idle(
        self, conn: websockets.sync.client.ClientConnection, stop: threading.Event
    ) -> None:
        """Close *conn* once no call has used it for ``_idle_timeout`` s.

        Sleeps until the exact deadline rather than polling, so the close
        lands on time; a call made meanwhile pushes the deadline out.
        """
        while True:
            with self._lock:
                if self._conn is not conn:
                    return  # already closed or replaced
                remaining = self._last_used + self._idle_timeout - time.monotonic()
                if remaining <= 0:
                    self._close_locked()
                    return
            if stop.wait(remaining):
                return


def _connect(
    fpga_id: int, api_key: str, api_url: str
) -> websockets.sync.client.ClientConnection:
    try:
        conn = websockets.sync.client.connect(
            _client._ws_url(api_url, fpga_id),
            additional_headers={"X-API-Key": api_key},
            open_timeout=_WS_OPEN_TIMEOUT,
        )
    except websockets.exceptions.InvalidStatus as exc:
        status = exc.response.status_code
        raise ConnectionError(
            f"Couldn't open a WebSocket to FPGA {fpga_id} (HTTP {status}). "
            f"Check the board is reserved by you: mrg status {fpga_id}. "
            f"{_HTTP_HINT}"
        ) from exc
    except (OSError, websockets.exceptions.InvalidHandshake) as exc:
        # OSError covers DNS failures, refused connections, and the
        # TimeoutError a stalled upgrade raises after _WS_OPEN_TIMEOUT.
        raise ConnectionError(
            f"Couldn't open a WebSocket to FPGA {fpga_id} ({exc}). {_HTTP_HINT}"
        ) from exc
    # websockets >= 16 warns when a connection is used outside
    # `with connect(...)`. This one deliberately outlives any single block --
    # _close_locked closes it -- so enter its context explicitly.
    return conn.__enter__()


def _closed_error(
    exc: websockets.exceptions.ConnectionClosed, fpga_id: int, what: str
) -> ConnectionError:
    """Explain a socket that closed while a call was waiting for its reply."""
    rcvd = exc.rcvd
    if rcvd is not None and rcvd.code == _CLOSE_BUSY:
        # The server never forwarded this call: it closes before reading.
        return ConnectionError(
            f"FPGA {fpga_id} is in use by another connection "
            f"(server: {rcvd.reason}). Close other scripts or terminals "
            "using this board, then try again."
        )
    detail = f" (server closed with {rcvd.code}: {rcvd.reason})" if rcvd else ""
    return ConnectionError(
        f"Connection to FPGA {fpga_id} lost during {what}{detail}. "
        "The operation may or may not have run on the FPGA."
    )


# What open_transport returns; App and the CLI only use read/write/close.
Transport = HttpTransport | WebSocketTransport
