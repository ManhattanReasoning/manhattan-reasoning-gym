"""SAT solver example using the Cloud FPGA SDK.

Run with:
    mrg run examples/sat_solver/client_sdk.py

Solves: (x1 ∨ x2) ∧ (¬x1 ∨ x3)
"""

import time

import manhattan_reasoning_gym

MAX_VARS = 10
MAX_CLAUSES = 20
CLAUSE_LEN = 10
LIT_BASE_WORD = 8  # first literal register is at word offset 8


class Regs(manhattan_reasoning_gym.cloud.RegisterMap):
    # word 0: W bit0=start (auto-clears), R bit0=done bit1=sat
    CTRL      = 0x0000
    N_VARS    = 0x0004   # word 1
    N_CLAUSES = 0x0008   # word 2
    MODEL     = 0x000C   # word 3 (valid when sat=1)
    CYCLES    = 0x0010   # word 4


def lit_addr(clause: int, slot: int) -> int:
    """Byte address of a literal register (clause c, slot s)."""
    word = LIT_BASE_WORD + clause * CLAUSE_LEN + slot
    return word * 4


def encode_lit(var: int, negated: bool) -> int:
    """Pack a literal as the register expects: bits[3:0]=var, bit4=neg, bit5=used."""
    return (1 << 5) | ((1 if negated else 0) << 4) | (var & 0xF)


app = manhattan_reasoning_gym.cloud.App(
    "sat_solver",
    design="examples/sat_solver/design.py",
    registers=Regs,
)

# FPGA BRUTE FORCE SAT SOLVER (for demonstration purposes only; not efficient!)

@app.local_entrypoint()
def main():
    with app:
        # Formula: (x1 ∨ x2) ∧ (¬x1 ∨ x3)   [0-based vars: 0=x1, 1=x2, 2=x3]
        clauses = [
            [(0, False), (1, False)],   # x1 ∨ x2
            [(0, True),  (2, False)],   # ¬x1 ∨ x3
        ]
        n_vars = 3
        n_clauses = len(clauses)

        app.write(Regs.N_VARS, n_vars)
        app.write(Regs.N_CLAUSES, n_clauses)

        for c, lits in enumerate(clauses):
            for slot, (var, neg) in enumerate(lits):
                app.write(lit_addr(c, slot), encode_lit(var, neg))

        app.write(Regs.CTRL, 1)  # start

        print("solving ...")
        while True:
            ctrl = app.read(Regs.CTRL)
            if ctrl & 1:  # done
                break
            time.sleep(0.005)

        cycles = app.read(Regs.CYCLES)
        sat = bool(ctrl & 2)

        if sat:
            model = app.read(Regs.MODEL)
            assignment = {f"x{i+1}": bool((model >> i) & 1) for i in range(n_vars)}
            print(f"SAT  ({cycles} cycles)  model={assignment}")
        else:
            print(f"UNSAT  ({cycles} cycles)")

        # UNSAT example: (x1) ∧ (¬x1)   — trivially contradictory
        clauses_unsat = [
            [(0, False)],   # x1
            [(0, True)],    # ¬x1
        ]
        n_vars_unsat = 1
        n_clauses_unsat = len(clauses_unsat)

        app.write(Regs.N_VARS, n_vars_unsat)
        app.write(Regs.N_CLAUSES, n_clauses_unsat)

        for c, lits in enumerate(clauses_unsat):
            for slot, (var, neg) in enumerate(lits):
                app.write(lit_addr(c, slot), encode_lit(var, neg))

        app.write(Regs.CTRL, 1)  # start

        print("solving (x1) ∧ (¬x1) ...")
        while True:
            ctrl = app.read(Regs.CTRL)
            if ctrl & 1:  # done
                break
            time.sleep(0.005)

        cycles = app.read(Regs.CYCLES)
        sat = bool(ctrl & 2)

        if sat:
            model = app.read(Regs.MODEL)
            assignment = {
                f"x{i+1}": bool((model >> i) & 1) for i in range(n_vars_unsat)
            }
            print(f"SAT  ({cycles} cycles)  model={assignment}")
        else:
            print(f"UNSAT  ({cycles} cycles)")
