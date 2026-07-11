"""Reference client for the SAT solver example.

Talks to a live FPGA running the generic Wishbone-bridge firmware. All SAT
knowledge lives here, client-side: formulas are encoded as register-level
Wishbone writes, the solver is started and polled, and the result registers
are read back -- all through the generic request/response wire protocol
defined in orchestrator/src/cloud_fpga_orchestrator/workers/protocol.py
(codec duplicated below so this example stays self-contained).

Usage:
    python client.py                # run built-in test formulas
    python client.py formula.cnf    # solve a DIMACS CNF file
    python client.py --host 192.168.1.101 --port 1234 formula.cnf
"""

import argparse
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum

from design import CLAUSE_LEN, LIT_BASE, MAX_CLAUSES, MAX_VARS

# Register byte offsets within the user design's Wishbone region
# (byte offset = 4 * word offset; see design.py register map).
REG_CTRL = 0x00  # W: start    R: bit0=done, bit1=sat
REG_NVARS = 0x04
REG_NCLAUSES = 0x08
REG_MODEL = 0x0C
REG_LITERALS = 4 * LIT_BASE  # 200 consecutive literal words

POLL_LIMIT = 1000  # done-bit polls before giving up


# ---------------------------------------------------------------------------
# Wire protocol codec -- must match workers/protocol.py exactly.
# ---------------------------------------------------------------------------


class WishboneOp(IntEnum):
    WRITE = 0x01
    READ = 0x02


class ResponseStatus(IntEnum):
    OK = 0x00
    ERROR = 0x01


@dataclass
class WishboneRequest:
    """A single Wishbone bus transaction to be sent to the FPGA.

    Wire format (big-endian):
        Byte 0:     opcode  (1 = write, 2 = read)
        Bytes 1-3:  length  (number of 32-bit words that follow)
        Bytes 4-7:  address (Wishbone byte offset)
        Bytes 8+:   data    (one 32-bit word per entry, write only)
    """

    op: WishboneOp
    address: int
    data: list[int] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        length = len(self.data)
        header = (
            struct.pack(">B", self.op)
            + struct.pack(">I", length)[1:]
            + struct.pack(">I", self.address)
        )
        words = struct.pack(f">{length}I", *self.data) if self.data else b""
        return header + words


@dataclass
class WishboneResponse:
    """The reply packet returned by the FPGA firmware.

    Wire format (big-endian):
        Byte 0:     status  (0x00 = ok, 0x01 = error)
        Bytes 1-3:  length  (number of 32-bit words that follow)
        Bytes 4+:   data    (one 32-bit word per entry)
    """

    status: ResponseStatus
    data: list[int]

    @property
    def ok(self) -> bool:
        return self.status == ResponseStatus.OK


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class FPGAConnection:
    """One TCP connection to the Wishbone-bridge firmware."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(timeout)
        self._sock.connect((host, port))

    def close(self):
        self._sock.close()

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("FPGA closed the connection")
            buf += chunk
        return buf

    def transact(self, request: WishboneRequest) -> WishboneResponse:
        """Send one request, block for its response."""
        self._sock.sendall(request.to_bytes())
        header = self._recv_exact(4)
        status = ResponseStatus(header[0])
        length = struct.unpack(">I", b"\x00" + header[1:4])[0]
        raw = self._recv_exact(length * 4)
        data = list(struct.unpack(f">{length}I", raw)) if length else []
        return WishboneResponse(status=status, data=data)

    def write(self, address: int, words: list[int]):
        resp = self.transact(
            WishboneRequest(op=WishboneOp.WRITE, address=address, data=words)
        )
        if not resp.ok:
            raise RuntimeError(f"write to {address:#x} failed")

    def read(self, address: int, count: int = 1) -> list[int]:
        resp = self.transact(
            WishboneRequest(op=WishboneOp.READ, address=address, data=[count])
        )
        if not resp.ok:
            raise RuntimeError(f"read of {count} words at {address:#x} failed")
        return resp.data


# ---------------------------------------------------------------------------
# SAT encoding -- formulas are lists of clauses; a clause is a list of
# signed ints, positive = variable, negative = negated (DIMACS convention).
# ---------------------------------------------------------------------------


def encode_literals(clauses: list[list[int]]) -> list[int]:
    """Pack a formula into the 200-word literal register block."""
    words = [0] * (MAX_CLAUSES * CLAUSE_LEN)
    for c, clause in enumerate(clauses):
        for slot, lit in enumerate(clause):
            var = abs(lit) - 1  # 0-based
            neg = 1 if lit < 0 else 0
            words[c * CLAUSE_LEN + slot] = (1 << 5) | (neg << 4) | var
    return words


def solve(conn: FPGAConnection, n_vars: int, clauses: list[list[int]]) -> dict:
    """Load a formula, run the solver, return the decoded result."""
    if not 1 <= n_vars <= MAX_VARS:
        raise ValueError(f"n_vars must be 1..{MAX_VARS}, got {n_vars}")
    if not 1 <= len(clauses) <= MAX_CLAUSES:
        raise ValueError(f"need 1..{MAX_CLAUSES} clauses, got {len(clauses)}")
    if any(len(cl) > CLAUSE_LEN for cl in clauses):
        raise ValueError(f"clauses are limited to {CLAUSE_LEN} literals")

    t0 = time.perf_counter()

    # One burst write loads (and implicitly clears) all literal registers.
    conn.write(REG_LITERALS, encode_literals(clauses))
    conn.write(REG_NVARS, [n_vars, len(clauses)])  # n_vars + n_clauses burst
    conn.write(REG_CTRL, [1])  # start (auto-clears in hardware)

    for _ in range(POLL_LIMIT):
        ctrl = conn.read(REG_CTRL)[0]
        if ctrl & 1:  # done
            break
    else:
        raise TimeoutError("solver never asserted done")

    elapsed_us = (time.perf_counter() - t0) * 1e6
    is_sat = bool((ctrl >> 1) & 1)

    result = {"result": "SAT" if is_sat else "UNSAT", "elapsed_us": elapsed_us}
    if is_sat:
        model = conn.read(REG_MODEL)[0]
        result["assignment"] = {
            f"x{i + 1}": bool((model >> i) & 1) for i in range(n_vars)
        }
    return result


def parse_dimacs(path: str) -> tuple[int, list[list[int]]]:
    """Parse a DIMACS CNF file. Returns (n_vars, clauses)."""
    n_vars = 0
    clauses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p cnf"):
                n_vars = int(line.split()[2])
                continue
            lits = [int(tok) for tok in line.split()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if lits:
                clauses.append(lits)
    return n_vars, clauses


def print_result(label: str, n_vars: int, clauses: list[list[int]], result: dict):
    print(f"[{label}] n_vars={n_vars}, n_clauses={len(clauses)}  "
          f"({result['elapsed_us']:.0f} us round trip)")
    print(f"  Result: {result['result']}")
    if "assignment" in result:
        ordered = sorted(result["assignment"].items(), key=lambda kv: int(kv[0][1:]))
        print("  Assignment:", " ".join(
            f"{k}={'T' if v else 'F'}" for k, v in ordered))
    print()


BUILTIN_TESTS = [
    ("4var_sat", 4, [[1, 2], [-1, 3], [-2, -3], [1, -2, 4]]),
    ("1var_unsat", 1, [[1], [-1]]),
    ("php32_unsat", 6, [
        [1, 2], [3, 4], [5, 6],
        [-1, -3], [-1, -5], [-3, -5],
        [-2, -4], [-2, -6], [-4, -6],
        [-1, -2], [-3, -4], [-5, -6],
    ]),
    ("6var_sat", 6, [[1, 2, 3], [-1, 4], [-2, 5], [-3, 6], [-4, -5], [-5, -6]]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cnf", nargs="?", help="DIMACS CNF file (omit for built-ins)")
    ap.add_argument("--host", default="192.168.1.101")
    ap.add_argument("--port", type=int, default=1234)
    args = ap.parse_args()

    conn = FPGAConnection(args.host, args.port)
    try:
        if args.cnf:
            n_vars, clauses = parse_dimacs(args.cnf)
            print_result(args.cnf, n_vars, clauses, solve(conn, n_vars, clauses))
        else:
            for label, n_vars, clauses in BUILTIN_TESTS:
                print_result(label, n_vars, clauses, solve(conn, n_vars, clauses))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
