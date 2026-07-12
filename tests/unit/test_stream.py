"""Tests for _client.Stream / App.stream() against a real local WebSocket
server (websockets.sync.server) -- no mocking of the websockets library
itself, so this exercises the actual wire round trip Stream depends on."""

from __future__ import annotations

import struct
import threading

import pytest
import websockets.sync.server

import manhattan_reasoning_gym as mrg
from manhattan_reasoning_gym._client import Stream, _ws_url


def _response_bytes(status: int, data: list[int]) -> bytes:
    header = struct.pack(">B", status) + struct.pack(">I", len(data))[1:]
    words = struct.pack(f">{len(data)}I", *data) if data else b""
    return header + words


class _RecordingServer:
    """A synchronous WS server that replies to each message with the next
    canned response, recording the requests and handshake it saw."""

    def __init__(self):
        self.requests: list[bytes] = []
        self.paths: list[str] = []
        self.headers: list[dict] = []
        self.replies: list[bytes] = []
        self._server = websockets.sync.server.serve(self._handle, "127.0.0.1", 0)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def _handle(self, ws) -> None:
        self.paths.append(ws.request.path)
        # Headers is case-insensitive; keep the object itself rather than a
        # plain dict so lookups like headers["X-API-Key"] still work.
        self.headers.append(ws.request.headers)
        for message in ws:
            self.requests.append(message)
            ws.send(self.replies.pop(0))

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
    srv = _RecordingServer()
    yield srv
    srv.close()


def test_ws_url_conversion():
    assert _ws_url("https://api.manhattanreasoning.com", 3) == (
        "wss://api.manhattanreasoning.com/fpga/3/stream"
    )
    assert _ws_url("http://127.0.0.1:9000", 0) == "ws://127.0.0.1:9000/fpga/0/stream"


def test_stream_write_sends_correct_frame_and_checks_status(server):
    server.replies = [_response_bytes(0x00, [])]
    with Stream(fpga_id=5, api_key="k", api_url=server.http_url) as s:
        s.write(0x10, 0xDEADBEEF)

    assert server.paths == ["/fpga/5/stream"]
    assert server.headers[0]["X-API-Key"] == "k"
    # opcode=WRITE(1), length=1, address=0x10, data=[0xDEADBEEF]
    assert server.requests == [
        bytes([0x01, 0x00, 0x00, 0x01])
        + struct.pack(">I", 0x10)
        + struct.pack(">I", 0xDEADBEEF)
    ]


def test_stream_write_fixed_address_sets_flag(server):
    server.replies = [_response_bytes(0x00, [])]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        s.write(0x20, [1, 2, 3], fixed_address=True)

    (raw,) = server.requests
    assert raw[0] == 0x01 | 0x80  # WRITE | FLAG_FIXED_ADDR


def test_stream_write_burst_list_single_frame(server):
    server.replies = [_response_bytes(0x00, [])]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        s.write(0x0, [1, 2, 3])

    assert len(server.requests) == 1  # one WS message for the whole burst


def test_stream_write_error_status_raises(server):
    server.replies = [_response_bytes(0x01, [])]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        with pytest.raises(RuntimeError, match="write to"):
            s.write(0x0, 1)


def test_stream_read_single_word_returns_int(server):
    server.replies = [_response_bytes(0x00, [0x2A])]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        assert s.read(0x8) == 0x2A


def test_stream_read_multi_word_returns_list(server):
    server.replies = [_response_bytes(0x00, [1, 2, 3])]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        assert s.read(0x8, count=3) == [1, 2, 3]


def test_stream_read_error_status_raises(server):
    server.replies = [_response_bytes(0x01, [])]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        with pytest.raises(RuntimeError, match="read of"):
            s.read(0x8)


def test_stream_reuses_one_connection_across_many_ops(server):
    server.replies = [_response_bytes(0x00, []) for _ in range(5)]
    with Stream(fpga_id=0, api_key="k", api_url=server.http_url) as s:
        for i in range(5):
            s.write(0x0, i)

    # Exactly one handshake for five ops -- the whole point of streaming.
    assert len(server.paths) == 1
    assert len(server.requests) == 5


def test_app_stream_returns_stream_and_talks_to_the_right_fpga(
    server, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MRG_API_KEY", raising=False)
    server.replies = [_response_bytes(0x00, [])]

    app = mrg.App("x", design="d.py", fpga_id=7, api_key="k", api_url=server.http_url)
    app._programmed = True  # skip the real build/program flow

    with app.stream() as s:
        assert isinstance(s, Stream)
        s.write(0x0, 1)

    assert server.paths == ["/fpga/7/stream"]
