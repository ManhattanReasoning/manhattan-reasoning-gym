"""Brute-force SAT solver -- Wishbone B4 slave (Amaranth HDL 0.5.x).

The primary research target workload for the Cloud_FPGA cluster: a hardware
boolean satisfiability solver exposed as a memory-mapped Wishbone register
file. Every clock cycle all clauses are evaluated combinationally in parallel
against the current candidate assignment while a binary counter sweeps the
assignment space. Worst case 2**MAX_VARS cycles (~20 us at 50 MHz).

Register map (32-bit word offsets; byte offset = 4 * word offset):

  word 0   W: bit0 = start (auto-clears next cycle)
           R: bit0 = done, bit1 = sat
  word 1   W: n_vars    (4 bits, 1..MAX_VARS)
  word 2   W: n_clauses (5 bits, 1..MAX_CLAUSES)
  word 3   R: model  -- bit i = value of variable i+1 (valid when sat=1)
  word 4   R: cycles -- clock cycles taken (20 bits, diagnostic)
  word 8 + c*CLAUSE_LEN + l
           W: literal for clause c, slot l:
              bits[3:0] = variable (0-based), bit4 = negated, bit5 = used

Writing 0 to a literal register clears its `used` bit, erasing the slot.
"""

from amaranth import Cat, Elaboratable, Module, Mux, Signal

MAX_VARS = 10
MAX_CLAUSES = 20
CLAUSE_LEN = 10

LIT_BASE = 8  # word offset of the first literal register


class BruteForceSAT(Elaboratable):
    """Brute-force SAT solver core.

    Inputs:
      start          1 bit   pulse high for 1 cycle to begin or restart
      n_vars         4 bits  number of variables (1 to MAX_VARS)
      n_clauses      5 bits  number of clauses (1 to MAX_CLAUSES)
      lit_var[c][l]  4 bits  which variable (0-based) is in clause c, slot l
      lit_neg[c][l]  1 bit   1 = this literal is negated
      lit_used[c][l] 1 bit   1 = this slot is occupied

    Outputs:
      done           1 bit   goes high when search is finished
      sat            1 bit   1 = satisfiable (only valid when done=1)
      model          10 bits satisfying assignment -- bit i = value of var i+1
      cycles         20 bits clock cycles taken (diagnostic)
    """

    def __init__(self):
        self.start = Signal()
        self.n_vars = Signal(range(MAX_VARS + 1))
        self.n_clauses = Signal(range(MAX_CLAUSES + 1))

        self.lit_var = [
            [Signal(range(MAX_VARS), name=f"lv_{c}_{lit}") for lit in range(CLAUSE_LEN)]
            for c in range(MAX_CLAUSES)
        ]
        self.lit_neg = [
            [Signal(name=f"ln_{c}_{lit}") for lit in range(CLAUSE_LEN)]
            for c in range(MAX_CLAUSES)
        ]
        self.lit_used = [
            [Signal(name=f"lu_{c}_{lit}") for lit in range(CLAUSE_LEN)]
            for c in range(MAX_CLAUSES)
        ]

        self.done = Signal()
        self.sat = Signal()
        self.model = Signal(MAX_VARS)
        self.cycles = Signal(20)

    def elaborate(self, platform):
        m = Module()

        assignment = Signal(MAX_VARS)
        running = Signal()
        done_r = Signal()
        sat_r = Signal()
        model_r = Signal(MAX_VARS)
        cycles_r = Signal(20)

        clause_sat = [Signal(name=f"cs_{c}") for c in range(MAX_CLAUSES)]

        for c in range(MAX_CLAUSES):
            lit_vals = []
            for lit in range(CLAUSE_LEN):
                var_val = Signal(name=f"vv_{c}_{lit}")
                lit_val = Signal(name=f"lval_{c}_{lit}")
                m.d.comb += var_val.eq(assignment.word_select(self.lit_var[c][lit], 1))
                m.d.comb += lit_val.eq(
                    Mux(self.lit_neg[c][lit], ~var_val, var_val)
                    & self.lit_used[c][lit]
                )
                lit_vals.append(lit_val)
            m.d.comb += clause_sat[c].eq(Cat(*lit_vals).any())

        all_sat = Signal()
        results_vec = Signal(MAX_CLAUSES)
        inactive_vec = Signal(MAX_CLAUSES)

        for ci in range(MAX_CLAUSES):
            m.d.comb += results_vec[ci].eq(clause_sat[ci])
            m.d.comb += inactive_vec[ci].eq(ci >= self.n_clauses)

        m.d.comb += all_sat.eq((results_vec | inactive_vec).all())

        max_assign = Signal(MAX_VARS)
        m.d.comb += max_assign.eq((1 << self.n_vars) - 1)

        with m.If(self.start):
            m.d.sync += [
                assignment.eq(0),
                done_r.eq(0),
                sat_r.eq(0),
                model_r.eq(0),
                cycles_r.eq(0),
                running.eq(1),
            ]
        with m.Elif(running):
            m.d.sync += cycles_r.eq(cycles_r + 1)
            with m.If(all_sat):
                m.d.sync += [
                    done_r.eq(1),
                    sat_r.eq(1),
                    model_r.eq(assignment),
                    running.eq(0),
                ]
            with m.Elif(assignment == max_assign):
                m.d.sync += [
                    done_r.eq(1),
                    sat_r.eq(0),
                    running.eq(0),
                ]
            with m.Else():
                m.d.sync += assignment.eq(assignment + 1)

        m.d.comb += [
            self.done.eq(done_r),
            self.sat.eq(sat_r),
            self.model.eq(model_r),
            self.cycles.eq(cycles_r),
        ]

        return m


class SATSlave(Elaboratable):
    """Wishbone B4 slave wrapping BruteForceSAT behind a register file.

    Conforms to the Cloud_FPGA user-design interface contract: classic
    Wishbone slave ports (cyc/stb/we/adr/dat_w/sel/dat_r/ack), 32-bit data,
    9-bit word address, registered 1-cycle ack.
    """

    def __init__(self):
        self.wb_cyc = Signal()
        self.wb_stb = Signal()
        self.wb_we = Signal()
        self.wb_adr = Signal(9)
        self.wb_dat_w = Signal(32)
        self.wb_sel = Signal(4)
        self.wb_dat_r = Signal(32)
        self.wb_ack = Signal()

    def elaborate(self, platform):
        m = Module()
        m.submodules.sat = sat = BruteForceSAT()

        # Control registers
        start_r = Signal()
        n_vars_r = Signal(range(MAX_VARS + 1))
        n_clauses_r = Signal(range(MAX_CLAUSES + 1))

        # Literal registers: packed as {used[5], neg[4], var[3:0]}
        lit_count = MAX_CLAUSES * CLAUSE_LEN
        lit_regs = [Signal(6, name=f"lit_{i}") for i in range(lit_count)]

        # Wire registers to BruteForceSAT
        m.d.comb += [
            sat.start.eq(start_r),
            sat.n_vars.eq(n_vars_r),
            sat.n_clauses.eq(n_clauses_r),
        ]
        for c in range(MAX_CLAUSES):
            for lit in range(CLAUSE_LEN):
                reg = lit_regs[c * CLAUSE_LEN + lit]
                m.d.comb += sat.lit_var[c][lit].eq(reg[:4])
                m.d.comb += sat.lit_neg[c][lit].eq(reg[4])
                m.d.comb += sat.lit_used[c][lit].eq(reg[5])

        # Auto-clear start after one cycle
        with m.If(start_r):
            m.d.sync += start_r.eq(0)

        # Ack: registered 1 cycle after cyc & stb, cleared otherwise
        with m.If(self.wb_cyc & self.wb_stb & ~self.wb_ack):
            m.d.sync += self.wb_ack.eq(1)
        with m.Else():
            m.d.sync += self.wb_ack.eq(0)

        # Write (fires on the cycle before ack)
        active_write = Signal()
        m.d.comb += active_write.eq(
            self.wb_cyc & self.wb_stb & self.wb_we & ~self.wb_ack
        )

        with m.If(active_write):
            with m.Switch(self.wb_adr):
                with m.Case(0):
                    m.d.sync += start_r.eq(self.wb_dat_w[0])
                with m.Case(1):
                    m.d.sync += n_vars_r.eq(self.wb_dat_w[:4])
                with m.Case(2):
                    m.d.sync += n_clauses_r.eq(self.wb_dat_w[:5])
                for i in range(lit_count):
                    with m.Case(LIT_BASE + i):
                        m.d.sync += lit_regs[i].eq(self.wb_dat_w[:6])

        # Read (combinatorial)
        with m.Switch(self.wb_adr):
            with m.Case(0):
                m.d.comb += self.wb_dat_r.eq(Cat(sat.done, sat.sat))
            with m.Case(3):
                m.d.comb += self.wb_dat_r.eq(sat.model)
            with m.Case(4):
                m.d.comb += self.wb_dat_r.eq(sat.cycles)
            with m.Default():
                m.d.comb += self.wb_dat_r.eq(0)

        return m
