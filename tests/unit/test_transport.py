"""Tests for _transport: choosing a transport, the HTTP job-path transport,
and WebSocketTransport against a real local WebSocket server
(websockets.sync.server) -- no mocking of the websockets library itself, so
these exercise the actual wire round trip the SDK depends on."""

from __future__ import annotations

import pytest

from manhattan_reasoning_gym import _transport

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
