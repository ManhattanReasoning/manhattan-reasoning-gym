"""Reference client for hello_wishbone: write a pattern, read it back.

The end-to-end smoke test for a live node: if this passes, the network
path, bridge firmware, bus wiring, and your design region all work.

Usage:
    python client.py [--host 192.168.1.101] [--port 1234] [--words 512]
"""

import argparse
import socket
import struct
import sys

OP_WRITE = 0x01
OP_READ = 0x02
STATUS_OK = 0x00


def encode_write(address: int, words: list[int]) -> bytes:
    header = struct.pack(">B", OP_WRITE) + struct.pack(">I", len(words))[1:]
    header += struct.pack(">I", address)
    return header + struct.pack(f">{len(words)}I", *words)


def encode_read(address: int, count: int) -> bytes:
    header = struct.pack(">B", OP_READ) + struct.pack(">I", 1)[1:]
    header += struct.pack(">I", address)
    return header + struct.pack(">I", count)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("FPGA closed the connection")
        buf += chunk
    return buf


def recv_response(sock: socket.socket) -> tuple[int, list[int]]:
    header = recv_exact(sock, 4)
    status = header[0]
    length = struct.unpack(">I", b"\x00" + header[1:4])[0]
    raw = recv_exact(sock, length * 4)
    data = list(struct.unpack(f">{length}I", raw)) if length else []
    return status, data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.1.101")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--words", type=int, default=512,
                    help="number of 32-bit words to test (max 512)")
    args = ap.parse_args()

    pattern = [(i * 0x01010101 + 0x5A5A) & 0xFFFFFFFF for i in range(args.words)]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(5.0)
    sock.connect((args.host, args.port))

    sock.sendall(encode_write(0, pattern))
    status, _ = recv_response(sock)
    if status != STATUS_OK:
        print("FAIL: write rejected")
        sys.exit(1)

    sock.sendall(encode_read(0, args.words))
    status, data = recv_response(sock)
    sock.close()

    if status != STATUS_OK:
        print("FAIL: read rejected")
        sys.exit(1)
    if data != pattern:
        diffs = sum(1 for a, b in zip(pattern, data) if a != b)
        print(f"FAIL: {diffs}/{args.words} words mismatched")
        sys.exit(1)

    print(f"OK: {args.words} words written and read back intact")


if __name__ == "__main__":
    main()
