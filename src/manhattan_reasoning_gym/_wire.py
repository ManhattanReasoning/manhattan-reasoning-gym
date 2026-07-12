"""Minimal Wishbone wire codec for the streaming client.

Mirrors orchestrator/src/cloud_fpga_orchestrator/workers/protocol.py (the
Manhattan-Reasoning-Cloud repo) so this package doesn't need to depend on the
private orchestrator package for the one thing App.stream() needs: encoding
requests and decoding responses for the raw byte-passthrough WebSocket relay
at /fpga/{id}/stream.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum


class WishboneOp(IntEnum):
    WRITE = 0x01
    READ = 0x02


# OR'd onto the opcode byte for a write: holds the address fixed across the
# burst instead of incrementing it per word. Must match FLAG_FIXED_ADDR in
# the orchestrator's protocol.py and OP_FLAG_FIXED_ADDR in
# firmware/sw/wishbone_tcp.c (mrg-sandbox repo).
FLAG_FIXED_ADDR = 0x80


class ResponseStatus(IntEnum):
    OK = 0x00
    ERROR = 0x01


@dataclass
class WishboneRequest:
    """A single Wishbone bus transaction, framed for the wire.

    Wire format (big-endian):
        Byte 0:     opcode  (1 = write, 2 = read; write may be OR'd with
                    FLAG_FIXED_ADDR)
        Bytes 1-3:  length  (number of 32-bit words that follow)
        Bytes 4-7:  address (Wishbone byte offset)
        Bytes 8+:   data
    """

    op: WishboneOp
    address: int
    data: list[int] = field(default_factory=list)
    count: int = 1
    fixed_address: bool = False

    def to_bytes(self) -> bytes:
        if self.op == WishboneOp.READ:
            length = 1
            words = struct.pack(">I", self.count)
            opcode = int(self.op)
        else:
            length = len(self.data)
            words = struct.pack(f">{length}I", *self.data) if self.data else b""
            opcode = int(self.op) | (FLAG_FIXED_ADDR if self.fixed_address else 0)
        header = (
            struct.pack(">B", opcode)
            + struct.pack(">I", length)[1:]
            + struct.pack(">I", self.address)
        )
        return header + words


@dataclass
class WishboneResponse:
    """The reply packet returned by the FPGA firmware."""

    status: ResponseStatus
    data: list[int]

    @property
    def ok(self) -> bool:
        return self.status == ResponseStatus.OK

    @classmethod
    def from_bytes(cls, raw: bytes) -> WishboneResponse:
        if len(raw) < 4:
            raise ValueError(f"Response too short: {len(raw)} bytes")
        status = ResponseStatus(raw[0])
        length = struct.unpack(">I", b"\x00" + raw[1:4])[0]
        expected = 4 + length * 4
        if len(raw) < expected:
            raise ValueError(
                f"Response truncated: expected {expected} bytes, got {len(raw)}"
            )
        data = list(struct.unpack(f">{length}I", raw[4:expected]))
        return cls(status=status, data=data)
