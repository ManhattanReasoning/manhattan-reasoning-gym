# sat_solver

Brute-force boolean satisfiability solver — the primary research target
workload. The solver evaluates all clauses combinationally in parallel each
clock cycle while a binary counter sweeps candidate assignments; worst case
2^10 = 1024 cycles ≈ 20 µs at 50 MHz.

## Contents

- `design.py` — the Amaranth design: `BruteForceSAT` core + `SATSlave`
  Wishbone B4 slave register file. Single file, ready to submit to the API.
- `client.py` — reference client: encodes CNF formulas into register-level
  Wishbone transactions using the generic wire protocol (see
  `orchestrator/.../workers/protocol.py`), runs the solver, decodes results.
- `tests/sim/` — Amaranth clock-cycle simulations of `SATSlave` (pytest, no hardware).
- `tests/unit/` — Python unit tests for design constants and Wishbone address map (`test_soc_parameters.py`) and bare-metal C framing tests compiled from `wishbone_tcp.c` (`test_network_stack.c`).

## Limits

| Parameter | Max | Set by |
|-----------|-----|--------|
| Variables | 10 | `MAX_VARS` (hardware size) |
| Clauses | 20 | `MAX_CLAUSES` |
| Literals per clause | 10 | `CLAUSE_LEN` |

## Register map

32-bit registers; byte offset = 4 × word offset, relative to the user
design's Wishbone region.

| Word offset | Byte offset | Access | Contents |
|------------|-------------|--------|----------|
| 0 | 0x000 | W | bit 0 = start (auto-clears next cycle) |
| 0 | 0x000 | R | bit 0 = done, bit 1 = sat |
| 1 | 0x004 | W | n_vars (4 bits) |
| 2 | 0x008 | W | n_clauses (5 bits) |
| 3 | 0x00C | R | model — bit *i* = value of variable *i+1* |
| 4 | 0x010 | R | cycles taken (20 bits, diagnostic) |
| 8 + c·10 + l | 0x020 + 4·(c·10+l) | W | literal for clause *c*, slot *l*: bits[3:0] = variable (0-based), bit 4 = negated, bit 5 = slot used |

Writing 0 to a literal register clears its `used` bit. The client clears the
whole literal block implicitly by always writing all 200 words in one burst.

## Solve sequence (what `client.py` does)

1. `WRITE` 200 words at byte offset 0x020 — the encoded literal block
2. `WRITE` 2 words at 0x004 — n_vars, n_clauses
3. `WRITE` 1 word at 0x000 — start
4. `READ` 0x000 until bit 0 (done) is set
5. If bit 1 (sat): `READ` 0x00C for the model

## Run

```sh
# Python unit + Amaranth simulation tests (no hardware)
pytest tests/

# Bare-metal C framing tests (no hardware, no cross-compiler)
gcc -std=c11 -Wall -Wextra -g tests/unit/test_network_stack.c -o /tmp/test_network_stack
/tmp/test_network_stack

# Against a live FPGA running the Wishbone-bridge firmware
python client.py                          # built-in test formulas
python client.py myformula.cnf            # DIMACS CNF file
python client.py --host 192.168.1.101 --port 1234 myformula.cnf
```
