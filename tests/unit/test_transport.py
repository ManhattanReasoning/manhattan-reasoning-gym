"""Tests for _transport: choosing a transport, the HTTP job-path transport,
and WebSocketTransport against a real local WebSocket server
(websockets.sync.server) -- no mocking of the websockets library itself, so
these exercise the actual wire round trip the SDK depends on."""

from __future__ import annotations

import struct
import threading
import time
import tomllib
from pathlib import Path

import pytest
import websockets.sync.server

import manhattan_reasoning_gym as mrg
from manhattan_reasoning_gym import _app as _app_mod
from manhattan_reasoning_gym import _transport
from manhattan_reasoning_gym._client import _ws_url

# -- resolve_transport ---------------------------------------------------------


def test_resolve_defaults_to_websocket(monkeypatch):
    monkeypatch.delenv("MRG_TRANSPORT", raising=False)
    assert _transport.resolve_transport() == "websocket"
    assert _transport.resolve_transport(None) == "websocket"


def test_resolve_reads_env(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", "http")
    assert _transport.resolve_transport() == "http"


def test_resolve_explicit_value_beats_env(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", "http")
    assert _transport.resolve_transport("websocket") == "websocket"


def test_resolve_ignores_case_and_whitespace(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", " HTTP ")
    assert _transport.resolve_transport() == "http"
    assert _transport.resolve_transport("WebSocket") == "websocket"


def test_resolve_empty_env_means_default(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", "")
    assert _transport.resolve_transport() == "websocket"


@pytest.mark.parametrize("bad", ["ftp", "ws", "https"])
def test_resolve_rejects_unknown_values(monkeypatch, bad):
    monkeypatch.delenv("MRG_TRANSPORT", raising=False)
    with pytest.raises(ValueError, match="expected 'websocket' or 'http'"):
        _transport.resolve_transport(bad)


def test_resolve_rejects_unknown_env_value(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError, match="carrier-pigeon"):
        _transport.resolve_transport()


# -- HttpTransport -------------------------------------------------------------


def test_http_transport_read_delegates_to_the_job_path(monkeypatch):
    seen = []

    def fake_read(fpga_id, api_key, address, count, api_url):
        seen.append((fpga_id, api_key, address, count, api_url))
        return [1, 2]

    monkeypatch.setattr(_transport._client, "read", fake_read)
    t = _transport.HttpTransport(3, "k", "https://api.example")
    assert t.read(0x8, count=2) == [1, 2]
    assert seen == [(3, "k", 0x8, 2, "https://api.example")]


def test_http_transport_write_delegates_to_the_job_path(monkeypatch):
    seen = []

    def fake_write(fpga_id, api_key, address, words, api_url, fixed_address=False):
        seen.append((fpga_id, api_key, address, words, api_url, fixed_address))

    monkeypatch.setattr(_transport._client, "write", fake_write)
    t = _transport.HttpTransport(3, "k", "https://api.example")
    t.write(0x10, [5, 6], fixed_address=True)
    assert seen == [(3, "k", 0x10, [5, 6], "https://api.example", True)]


def test_http_transport_close_is_a_no_op():
    t = _transport.HttpTransport(3, "k", "https://api.example")
    t.close()
    t.close()


# -- a local WebSocket server standing in for the orchestrator's /stream ------


def _response_bytes(status: int, data: list[int]) -> bytes:
    header = struct.pack(">B", status) + struct.pack(">I", len(data))[1:]
    words = struct.pack(f">{len(data)}I", *data) if data else b""
    return header + words


def _default_reply(message: bytes) -> bytes:
    """OK for a write; for a read, echo back the addresses that were read."""
    if message[0] & 0x7F == 0x02:  # READ
        address = struct.unpack(">I", message[4:8])[0]
        count = struct.unpack(">I", message[8:12])[0]
        return _response_bytes(0x00, [address + 4 * i for i in range(count)])
    return _response_bytes(0x00, [])


class _Server:
    """A real local WebSocket server. Knobs, set before the client acts:

    - replies: canned reply frames, used first-in first-out; when empty,
      _default_reply answers.
    - reject_status: refuse the handshake with this HTTP status.
    - close_with: on the next request, close with (code, reason) instead of
      replying.
    - close_after_reply: close normally right after replying.
    - delay: seconds to wait before replying.
    """

    def __init__(self):
        self.requests: list[bytes] = []
        self.paths: list[str] = []
        self.headers: list = []
        self.replies: list = []
        self.reject_status: int | None = None
        self.close_with: tuple[int, str] | None = None
        self.close_after_reply = False
        self.delay = 0.0
        self.ended = 0  # connections whose handler has finished
        self._ended_lock = threading.Lock()
        self._server = websockets.sync.server.serve(
            self._handle, "127.0.0.1", 0, process_request=self._process_request
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def _process_request(self, connection, request):
        if self.reject_status is not None:
            return connection.respond(self.reject_status, "rejected\n")
        return None

    def _handle(self, ws) -> None:
        self.paths.append(ws.request.path)
        # Headers is case-insensitive; keep the object itself rather than a
        # plain dict so lookups like headers["X-API-Key"] still work.
        self.headers.append(ws.request.headers)
        try:
            for message in ws:
                self.requests.append(message)
                if self.close_with is not None:
                    ws.close(*self.close_with)
                    return
                if self.delay:
                    time.sleep(self.delay)
                reply = self.replies.pop(0) if self.replies else None
                ws.send(reply if reply is not None else _default_reply(message))
                if self.close_after_reply:
                    ws.close()
                    return
        finally:
            with self._ended_lock:
                self.ended += 1

    @property
    def port(self) -> int:
        return self._server.socket.getsockname()[1]

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2)


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.close()


@pytest.fixture
def transports():
    """Make WebSocketTransports that are always closed after the test."""
    made = []

    def make(server: _Server, fpga_id: int = 0) -> _transport.WebSocketTransport:
        t = _transport.WebSocketTransport(fpga_id, "k", server.http_url)
        made.append(t)
        return t

    yield make
    for t in made:
        t.close()


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# -- WebSocketTransport: frames and connection reuse ----------------------------


def test_ws_url_conversion():
    assert _ws_url("https://api.manhattanreasoning.com", 3) == (
        "wss://api.manhattanreasoning.com/fpga/3/stream"
    )
    assert _ws_url("http://127.0.0.1:9000", 0) == "ws://127.0.0.1:9000/fpga/0/stream"


def test_open_transport_picks_the_class(monkeypatch):
    monkeypatch.delenv("MRG_TRANSPORT", raising=False)
    ws = _transport.open_transport(None, 3, "k", "https://api.example")
    http = _transport.open_transport("http", 3, "k", "https://api.example")
    assert isinstance(ws, _transport.WebSocketTransport)
    assert isinstance(http, _transport.HttpTransport)


def test_open_transport_validates_kind():
    with pytest.raises(ValueError):
        _transport.open_transport("ftp", 3, "k", "https://api.example")


def test_nothing_connects_before_the_first_call(server, transports):
    t = transports(server)
    time.sleep(0.05)
    assert server.paths == []
    assert t._conn is None


def test_write_sends_correct_frame_to_the_right_fpga(server, transports):
    transports(server, fpga_id=5).write(0x10, [0xDEADBEEF])
    assert server.paths == ["/fpga/5/stream"]
    assert server.headers[0]["X-API-Key"] == "k"
    # opcode=WRITE(1), length=1, address=0x10, data=[0xDEADBEEF]
    assert server.requests == [
        bytes([0x01, 0x00, 0x00, 0x01])
        + struct.pack(">I", 0x10)
        + struct.pack(">I", 0xDEADBEEF)
    ]


def test_write_fixed_address_sets_flag(server, transports):
    transports(server).write(0x20, [1, 2, 3], fixed_address=True)
    (raw,) = server.requests
    assert raw[0] == 0x01 | 0x80  # WRITE | FLAG_FIXED_ADDR


def test_burst_write_is_one_message(server, transports):
    transports(server).write(0x0, [1, 2, 3])
    assert len(server.requests) == 1


def test_read_returns_a_list_of_words(server, transports):
    server.replies = [_response_bytes(0x00, [0x2A])]
    t = transports(server)
    assert t.read(0x8) == [0x2A]
    assert t.read(0x8, count=3) == [0x8, 0xC, 0x10]  # default reply echoes


def test_write_error_status_raises(server, transports):
    server.replies = [_response_bytes(0x01, [])]
    with pytest.raises(RuntimeError, match="write to 0x0 failed"):
        transports(server).write(0x0, [1])


def test_read_error_status_raises(server, transports):
    server.replies = [_response_bytes(0x01, [])]
    with pytest.raises(RuntimeError, match="read of 1 words at 0x8 failed"):
        transports(server).read(0x8)


def test_many_calls_share_one_connection(server, transports):
    t = transports(server)
    for i in range(5):
        t.write(0x0, [i])
    # Exactly one handshake for five ops -- the whole point of the socket.
    assert len(server.paths) == 1
    assert len(server.requests) == 5


def test_close_closes_the_socket_and_is_idempotent(server, transports):
    t = transports(server)
    t.write(0x0, [1])
    t.close()
    t.close()
    assert t._conn is None
    assert _wait_until(lambda: server.ended == 1)


def test_call_after_close_opens_a_new_socket(server, transports):
    t = transports(server)
    t.write(0x0, [1])
    t.close()
    t.write(0x0, [2])
    assert len(server.paths) == 2


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_long_lived_socket_raises_no_deprecation_warning(server, transports):
    # websockets >= 16 warns when a connection is used outside
    # `with connect(...)`; ours outlives any one block, so it must opt in.
    t = transports(server)
    t.write(0x0, [1])
    assert t.read(0x0) == [0x0]


# -- WebSocketTransport: errors -------------------------------------------------


def test_rejected_handshake_names_the_status_and_the_fixes(server, transports):
    server.reject_status = 403
    with pytest.raises(ConnectionError) as info:
        transports(server, fpga_id=3).write(0x0, [1])
    msg = str(info.value)
    assert "Couldn't open a WebSocket to FPGA 3 (HTTP 403)" in msg
    assert "mrg status 3" in msg
    assert 'transport="http"' in msg
    assert "MRG_TRANSPORT=http" in msg
    assert server.requests == []


@pytest.mark.parametrize(
    "error", [OSError("connection refused"), TimeoutError("timed out in handshake")]
)
def test_network_failure_on_open_names_the_fix(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_transport.websockets.sync.client, "connect", fail)
    t = _transport.WebSocketTransport(3, "k", "https://api.example")
    with pytest.raises(ConnectionError) as info:
        t.read(0x0)
    msg = str(info.value)
    assert "Couldn't open a WebSocket to FPGA 3" in msg
    assert str(error) in msg
    assert "MRG_TRANSPORT=http" in msg
    assert info.value.__cause__ is error


def test_busy_board_is_reported_as_in_use(server, transports):
    server.close_with = (4409, "FPGA is busy with a run job")
    with pytest.raises(ConnectionError) as info:
        transports(server, fpga_id=3).write(0x0, [1])
    msg = str(info.value)
    assert "FPGA 3 is in use by another connection" in msg
    assert "FPGA is busy with a run job" in msg


@pytest.mark.parametrize(
    "code, reason", [(4502, "FPGA connection failed: timed out"), (1011, "oops")]
)
def test_drop_during_a_call_says_it_may_have_run(server, transports, code, reason):
    server.close_with = (code, reason)
    t = transports(server, fpga_id=3)
    with pytest.raises(ConnectionError) as info:
        t.write(0x10, [1])
    msg = str(info.value)
    assert "Connection to FPGA 3 lost during write to 0x10" in msg
    assert f"server closed with {code}: {reason}" in msg
    assert "may or may not have run" in msg
    assert t._conn is None


def test_drop_without_a_close_frame_says_it_may_have_run():
    exc = _transport.websockets.exceptions.ConnectionClosed(None, None)
    err = _transport._closed_error(exc, 3, "read of 1 words at 0x8")
    assert str(err) == (
        "Connection to FPGA 3 lost during read of 1 words at 0x8. "
        "The operation may or may not have run on the FPGA."
    )


def test_next_call_after_a_drop_opens_a_fresh_socket(server, transports):
    server.close_with = (4502, "FPGA connection failed")
    t = transports(server)
    with pytest.raises(ConnectionError):
        t.write(0x0, [1])
    server.close_with = None
    t.write(0x0, [2])
    assert len(server.paths) == 2


def test_socket_closed_while_idle_is_replaced_silently(server, transports):
    server.close_after_reply = True
    t = transports(server)
    t.write(0x0, [1])
    # The server hung up after replying; wait for that close to finish so the
    # client's socket is no longer open when the next call goes out.
    assert _wait_until(lambda: server.ended == 1)
    server.close_after_reply = False
    t.write(0x0, [2])
    assert len(server.paths) == 2
    # Each call reached the server exactly once.
    assert [struct.unpack(">I", r[8:12])[0] for r in server.requests] == [1, 2]


def test_text_frame_reply_raises(server, transports):
    server.replies = ["not a frame"]
    with pytest.raises(RuntimeError, match="unexpected text frame"):
        transports(server).read(0x0)


# -- WebSocketTransport: idle close and threads ---------------------------------


@pytest.fixture
def short_idle(monkeypatch):
    monkeypatch.setattr(_transport, "_WS_IDLE_TIMEOUT", 0.2)


def test_idle_socket_closes_itself_and_reopens_on_next_call(
    server, transports, short_idle
):
    t = transports(server)
    t.write(0x0, [1])
    assert _wait_until(lambda: server.ended == 1)  # closed by the idle timer
    assert t._conn is None
    t.write(0x0, [2])
    assert len(server.paths) == 2


def test_calls_keep_pushing_the_idle_deadline_out(server, transports, short_idle):
    t = transports(server)
    for i in range(10):  # 0.5 s of activity, never 0.2 s idle
        t.write(0x0, [i])
        time.sleep(0.05)
    assert len(server.paths) == 1
    assert server.ended == 0


def test_idle_timer_never_closes_during_a_slow_call(server, transports, short_idle):
    server.delay = 0.5  # the reply takes longer than the idle timeout
    t = transports(server)
    assert t.read(0x8) == [0x8]
    assert server.ended == 0


def test_close_stops_the_idle_watcher(server, transports):
    t = transports(server)
    t.write(0x0, [1])
    watcher = t._watcher
    assert watcher.is_alive()
    t.close()
    assert not watcher.is_alive()


def test_idle_watcher_does_not_keep_the_process_alive(server, transports):
    t = transports(server)
    t.write(0x0, [1])
    assert t._watcher.daemon


def test_threads_sharing_a_transport_get_their_own_replies(server, transports):
    t = transports(server)
    results: dict[int, list[int]] = {0: [], 1: []}

    def worker(n: int) -> None:
        base = 0x1000 * (n + 1)
        for i in range(50):
            results[n].append(t.read(base + 4 * i)[0])

    threads = [threading.Thread(target=worker, args=(n,)) for n in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)
    assert results[0] == [0x1000 + 4 * i for i in range(50)]
    assert results[1] == [0x2000 + 4 * i for i in range(50)]
    assert len(server.paths) == 1


# -- App ------------------------------------------------------------------------


def _make_app(monkeypatch, tmp_path, *, api_url, **kwargs):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MRG_API_KEY", raising=False)
    monkeypatch.delenv("MRG_TRANSPORT", raising=False)
    app = mrg.App(
        "x", design="d.py", fpga_id=7, api_key="k", api_url=api_url, **kwargs
    )
    app._programmed = True  # skip the real build/program flow
    return app


@pytest.fixture
def no_release(monkeypatch):
    """Stand in for the server's session release (the board reset job)."""
    calls = []
    monkeypatch.setattr(
        _app_mod._client, "release_session",
        lambda *args: calls.append(args) or "job-1",
    )
    return calls


def test_app_uses_a_websocket_by_default(
    server, monkeypatch, tmp_path, no_release
):
    app = _make_app(monkeypatch, tmp_path, api_url=server.http_url)
    with app:
        app.write(0x0, 1)
        app.write(0x4, [2, 3])
        assert app.read(0x8) == 0x8  # one word comes back as an int
        assert app.read(0x8, count=2) == [0x8, 0xC]
    assert server.paths == ["/fpga/7/stream"]  # one socket for all four calls
    assert len(server.requests) == 4


def test_app_http_transport_never_opens_a_socket(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        _transport._client, "read", lambda *a, **k: calls.append(("read", a)) or [42]
    )
    monkeypatch.setattr(
        _transport._client, "write", lambda *a, **k: calls.append(("write", a, k))
    )

    def no_socket(*args, **kwargs):
        raise AssertionError("opened a WebSocket")

    monkeypatch.setattr(_transport.websockets.sync.client, "connect", no_socket)
    app = _make_app(
        monkeypatch, tmp_path, api_url="https://api.example", transport="http"
    )
    assert app.read(0x8) == 42
    app.write(0x10, 5, fixed_address=True)
    assert calls[0] == ("read", (7, "k", 0x8, 1, "https://api.example"))
    assert calls[1] == (
        "write", (7, "k", 0x10, [5], "https://api.example"), {"fixed_address": True}
    )


def test_app_transport_from_env(monkeypatch):
    monkeypatch.setenv("MRG_TRANSPORT", "http")
    assert mrg.App("x", design="d.py", api_key="k").transport == "http"


def test_app_rejects_unknown_transport(monkeypatch):
    monkeypatch.delenv("MRG_TRANSPORT", raising=False)
    with pytest.raises(ValueError, match="expected 'websocket' or 'http'"):
        mrg.App("x", design="d.py", api_key="k", transport="ftp")


def test_release_closes_the_socket_before_releasing_the_board(
    monkeypatch, tmp_path
):
    app = _make_app(monkeypatch, tmp_path, api_url="https://api.example")
    order = []

    class _FakeConnection:
        def close(self):
            order.append("close")

    app._connection = _FakeConnection()
    monkeypatch.setattr(
        _app_mod._client, "release_session",
        lambda *args: order.append("release") or "job-1",
    )
    assert app.release() == "job-1"
    assert order == ["close", "release"]
    assert app._connection is None


def test_release_without_a_socket_is_fine(monkeypatch, tmp_path, no_release):
    app = _make_app(monkeypatch, tmp_path, api_url="https://api.example")
    app.release()
    app.release()
    assert len(no_release) == 2


def test_leaving_with_app_closes_the_socket(
    server, monkeypatch, tmp_path, no_release
):
    app = _make_app(monkeypatch, tmp_path, api_url=server.http_url)
    with app:
        app.write(0x0, 1)
    assert app._connection is None
    assert _wait_until(lambda: server.ended == 1)


def test_app_reused_after_release_talks_to_the_new_board(
    server, monkeypatch, tmp_path, no_release
):
    app = _make_app(monkeypatch, tmp_path, api_url=server.http_url)
    with app:
        app.write(0x0, 1)
    # A second `with app:` reprograms first and may land on another board.
    app._programmed, app.fpga_id = True, 8
    with app:
        app.write(0x0, 2)
    assert server.paths == ["/fpga/7/stream", "/fpga/8/stream"]


def test_stream_is_gone_from_the_public_api():
    assert not hasattr(mrg.App, "stream")
    assert not hasattr(mrg.cloud, "Stream")
    assert "Stream" not in mrg.cloud.__all__


def test_interrupted_call_drops_the_socket_so_replies_stay_paired(
    server, transports
):
    t = transports(server)
    t.read(0x1000)  # opens the socket
    conn = t._conn
    real_recv = conn.recv

    def interrupted_recv(*args, **kwargs):
        conn.recv = real_recv
        raise KeyboardInterrupt  # e.g. Ctrl-C while waiting for the reply

    conn.recv = interrupted_recv
    with pytest.raises(KeyboardInterrupt):
        t.read(0x2000)
    # The interrupted read's reply can still arrive on that socket; the relay
    # pairs replies with requests by order, so the next call needs a new one.
    assert t.read(0x3000) == [0x3000]
    assert len(server.paths) == 2


def test_websockets_floor_has_sync_keepalive():
    # The sync client only sends keep-alive pings from websockets 15.0; below
    # that, a silently dead network can hang a call's recv() forever.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert "websockets>=15.0" in deps
