"""Wire codec tests for _wire.py -- mirrors the orchestrator's protocol.py
tests (Manhattan-Reasoning-Cloud repo) since this is a deliberate duplicate
kept in sync by hand, the same way examples/sat_solver/client.py is."""

import struct

import pytest

from manhattan_reasoning_gym._wire import (
    FLAG_FIXED_ADDR,
    ResponseStatus,
    WishboneOp,
    WishboneRequest,
    WishboneResponse,
)


class TestWishboneRequestToBytes:
    def test_write_opcode_byte_default(self):
        raw = WishboneRequest(op=WishboneOp.WRITE, address=0x0, data=[0x1]).to_bytes()
        assert raw[0] == WishboneOp.WRITE

    def test_fixed_address_sets_flag_bit(self):
        raw = WishboneRequest(
            op=WishboneOp.WRITE, address=0x0, data=[0x1], fixed_address=True
        ).to_bytes()
        assert raw[0] == WishboneOp.WRITE | FLAG_FIXED_ADDR

    def test_fixed_address_ignored_on_read(self):
        raw = WishboneRequest(
            op=WishboneOp.READ, address=0x0, count=1, fixed_address=True
        ).to_bytes()
        assert raw[0] == WishboneOp.READ

    def test_write_length_and_data(self):
        raw = WishboneRequest(
            op=WishboneOp.WRITE, address=0x4, data=[0xDEADBEEF, 0xCAFEBABE]
        ).to_bytes()
        assert raw[1:4] == b"\x00\x00\x02"
        assert struct.unpack(">I", raw[8:12])[0] == 0xDEADBEEF
        assert struct.unpack(">I", raw[12:16])[0] == 0xCAFEBABE

    def test_read_carries_count_word(self):
        raw = WishboneRequest(op=WishboneOp.READ, address=0x1000, count=3).to_bytes()
        assert len(raw) == 12
        assert struct.unpack(">I", raw[8:12])[0] == 3


class TestWishboneResponseFromBytes:
    def test_ok_status_no_data(self):
        resp = WishboneResponse.from_bytes(bytes([0x00, 0x00, 0x00, 0x00]))
        assert resp.status == ResponseStatus.OK
        assert resp.ok is True
        assert resp.data == []

    def test_error_status(self):
        resp = WishboneResponse.from_bytes(bytes([0x01, 0x00, 0x00, 0x00]))
        assert resp.ok is False

    def test_data_words_decoded(self):
        raw = bytes([0x00, 0x00, 0x00, 0x02]) + struct.pack(">II", 0xAA, 0xBB)
        assert WishboneResponse.from_bytes(raw).data == [0xAA, 0xBB]

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            WishboneResponse.from_bytes(b"\x00\x00\x00")

    def test_truncated_payload_raises(self):
        raw = bytes([0x00, 0x00, 0x00, 0x02]) + b"\xAA\xBB\xCC\xDD"
        with pytest.raises(ValueError, match="truncated"):
            WishboneResponse.from_bytes(raw)
