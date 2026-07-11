"""FFN accelerator as a Cloud-FPGA user_design (self-contained, single file).

A streaming, tiled, INT8 BERT feed-forward engine -- `requant2(GELU(requant1(
x.W1)).W2)` -- wrapped behind the base SoC's 2 KB Wishbone window so it builds +
runs on the live board through the existing MMIO path (no DMA, no SoC changes).
The engine does all the tiling, requantization, and GELU internally; the host
just streams operands in, pulses start, and reads the int8 result out.

This file is deliberately standalone (only Amaranth) because the orchestrator
uploads the single design file to the build host. The integer math is mirrored
bit-for-bit by `golden.py`, which the sim tests and SDK client check against.

Pipeline:

    x ─▶ MatMul-1 ─▶ Requant ─▶ GELU LUT ─▶ MatMul-2 ─▶ Requant ─▶ y   (all int8 I/O)
         x.W1        i32→i8      i8→i8       .W2         i32→i8

Operands are far larger than the 512-word window, so access is *windowed*: a
TARGET register selects a buffer (x / W1 / W2 / GELU table), a PTR register sets
a base index, and the DATA window (word 16+) reads/writes ``buffer[PTR + off]``.
With tile=4 every packed word (4 int8 lanes) is exactly one 32-bit MMIO word.

Register map (byte offset = 4 * word):

  word 0   CTRL    W bit0=start (auto-clears)   R bit0=done, bit1=busy
  word 1   TARGET  W 0=x, 1=W1, 2=W2, 3=GELU table
  word 2   PTR     W base index into the selected buffer
  word 3   MULT1   W requant-1 fixed-point multiplier
  word 4   SHIFT1  W requant-1 shift
  word 5   MULT2   W requant-2 multiplier
  word 6   SHIFT2  W requant-2 shift
  word 7   ID_M    R hardware M (client verifies dims before loading)
  word 8   ID_K1   R hardware K1
  word 9   ID_K2   R hardware K2 (hidden)
  word 10  ID_N    R hardware N
  word 11  ID_TILE R hardware tile size
  word 16+ DATA    W -> selected buffer[PTR+off]    R -> result C[PTR+off]
"""

from amaranth import Array, Cat, Elaboratable, Module, Mux, Signal, signed, unsigned
from amaranth.lib.memory import Memory

# ── element / fixed-point parameters (mirrored in golden.py) ──────────────────

IN_W = 8  # signed int8 activations and weights, values in [-127, 127]
ACC_W = 32  # signed int32 dot-product accumulator
INT8_MAX = 127  # symmetric range; -128 unused
REQUANT_MULT_W = 32
TILE_M = TILE_N = 8  # defaults for standalone instantiation / tests
_PROD_W = ACC_W + REQUANT_MULT_W + 1
SHIFT_W = 6


# ── MAC grid: tile_m x tile_n output-stationary INT8->INT32 MACs ──────────────


class MacArray(Elaboratable):
    """Every enabled cycle, fan one A column + one B row across tile_m*tile_n MACs.

    Contract per output tile: pulse ``clear`` (zero accumulators), then ``en`` for
    K cycles presenting ``a_col = A[:,k]`` / ``b_row = B[k,:]``; afterwards
    ``c[i*tile_n+j]`` holds ``sum_k A[i,k]*B[k,j]``.
    """

    def __init__(self, tile_m=TILE_M, tile_n=TILE_N):
        self.tile_m, self.tile_n = tile_m, tile_n
        self.clear = Signal()
        self.en = Signal()
        self.a_col = [Signal(signed(IN_W), name=f"a_{i}") for i in range(tile_m)]
        self.b_row = [Signal(signed(IN_W), name=f"b_{j}") for j in range(tile_n)]
        self.c = [
            Signal(signed(ACC_W), name=f"c_{i}_{j}")
            for i in range(tile_m)
            for j in range(tile_n)
        ]

    def elaborate(self, platform):
        m = Module()
        for i in range(self.tile_m):
            for j in range(self.tile_n):
                acc = self.c[i * self.tile_n + j]
                prod = self.a_col[i] * self.b_row[j]  # int8*int8, one DSP
                with m.If(self.clear):
                    m.d.sync += acc.eq(0)
                with m.Elif(self.en):
                    m.d.sync += acc.eq(acc + prod)
        return m


# ── fixed-point requantizer: int32 acc -> int8 (gemmlowp style) ───────────────


class Requant(Elaboratable):
    """out = clamp((acc*mult + (1<<(shift-1))) >> shift, -127, 127). 1-cycle latency."""

    def __init__(self):
        self.in_valid = Signal()
        self.acc = Signal(signed(ACC_W))
        self.mult = Signal(REQUANT_MULT_W)
        self.shift = Signal(SHIFT_W)
        self.out_valid = Signal()
        self.out = Signal(signed(IN_W))

    def elaborate(self, platform):
        m = Module()
        prod = Signal(signed(_PROD_W))
        bias = Signal(signed(_PROD_W))
        shifted = Signal(signed(_PROD_W))
        m.d.comb += [
            prod.eq(self.acc * self.mult),
            bias.eq((1 << self.shift) >> 1),
            shifted.eq((prod + bias) >> self.shift),  # signed -> arithmetic shift
        ]
        m.d.sync += self.out_valid.eq(self.in_valid)
        with m.If(shifted > INT8_MAX):
            m.d.sync += self.out.eq(INT8_MAX)
        with m.Elif(shifted < -INT8_MAX):
            m.d.sync += self.out.eq(-INT8_MAX)
        with m.Else():
            m.d.sync += self.out.eq(shifted)
        return m


# ── GELU as a runtime-loadable int8 -> int8 lookup ROM ────────────────────────

GELU_DEPTH = 1 << IN_W


class Gelu(Elaboratable):
    """256-entry int8->int8 GELU table, indexed by the two's-complement input."""

    def __init__(self, init=None):
        self._init = list(init) if init is not None else [0] * GELU_DEPTH
        self.load_en = Signal()
        self.load_addr = Signal(IN_W)
        self.load_data = Signal(signed(IN_W))
        self.in_valid = Signal()
        self.in_code = Signal(signed(IN_W))
        self.out_valid = Signal()
        self.out = Signal(signed(IN_W))

    def elaborate(self, platform):
        m = Module()
        m.submodules.mem = mem = Memory(
            shape=signed(IN_W), depth=GELU_DEPTH, init=self._init
        )
        rp, wp = mem.read_port(), mem.write_port()
        m.d.comb += [
            wp.addr.eq(self.load_addr),
            wp.data.eq(self.load_data),
            wp.en.eq(self.load_en),
            rp.addr.eq(self.in_code),
            self.out.eq(rp.data),
        ]
        m.d.sync += self.out_valid.eq(self.in_valid)
        return m


# ── tiled INT8 matmul engine: C[M,N] = requant(A[M,K] @ B[K,N]) ────────────────


class MatMulEngine(Elaboratable):
    """Walks the output in tile_m x tile_n tiles; streams K depth-steps through the
    MAC grid from packed BRAM, then requantizes each tile to int8 and writes it out.

    Output is emitted in the same packed A-layout the engine consumes (tile_m lanes,
    indexed [mi, col]) so one engine's C feeds straight into the next's A.
    """

    def __init__(self, m_dim, k_dim, n_dim, tile_m, tile_n):
        assert m_dim % tile_m == 0 and n_dim % tile_n == 0
        self.M, self.K, self.N = m_dim, k_dim, n_dim
        self.tile_m, self.tile_n = tile_m, tile_n
        self.MTM, self.NTN = m_dim // tile_m, n_dim // tile_n
        self.a_depth = self.MTM * k_dim
        self.b_depth = k_dim * self.NTN
        self.c_depth = self.MTM * n_dim

        self.a_we = Signal()
        self.a_waddr = Signal(range(self.a_depth))
        self.a_wdata = Signal(tile_m * IN_W)
        self.b_we = Signal()
        self.b_waddr = Signal(range(self.b_depth))
        self.b_wdata = Signal(tile_n * IN_W)
        self.mult = Signal(32)
        self.shift = Signal(6)
        self.start = Signal()
        self.busy = Signal()
        self.done = Signal()
        self.c_raddr = Signal(range(self.c_depth))
        self.c_rdata = Signal(tile_m * IN_W)

    def elaborate(self, platform):
        m = Module()
        TM, TN, K = self.tile_m, self.tile_n, self.K
        m.submodules.mac = mac = MacArray(TM, TN)
        m.submodules.rq = rq = Requant()
        m.submodules.amem = amem = Memory(
            shape=unsigned(TM * IN_W), depth=self.a_depth, init=[]
        )
        m.submodules.bmem = bmem = Memory(
            shape=unsigned(TN * IN_W), depth=self.b_depth, init=[]
        )
        m.submodules.cmem = cmem = Memory(
            shape=unsigned(TM * IN_W), depth=self.c_depth, init=[]
        )
        a_rp, a_wp = amem.read_port(), amem.write_port()
        b_rp, b_wp = bmem.read_port(), bmem.write_port()
        c_rp, c_wp = cmem.read_port(), cmem.write_port()

        m.d.comb += [
            a_wp.addr.eq(self.a_waddr),
            a_wp.data.eq(self.a_wdata),
            a_wp.en.eq(self.a_we),
            b_wp.addr.eq(self.b_waddr),
            b_wp.data.eq(self.b_wdata),
            b_wp.en.eq(self.b_we),
            c_rp.addr.eq(self.c_raddr),
            self.c_rdata.eq(c_rp.data),
            rq.mult.eq(self.mult),
            rq.shift.eq(self.shift),
        ]

        mi = Signal(range(self.MTM + 1))
        nj = Signal(range(self.NTN + 1))
        req_k = Signal(range(K + 1))
        en_d = Signal()
        m.d.comb += [a_rp.addr.eq(mi * K + req_k), b_rp.addr.eq(req_k * self.NTN + nj)]
        for i in range(TM):
            m.d.comb += mac.a_col[i].eq(
                a_rp.data[i * IN_W : (i + 1) * IN_W].as_signed()
            )
        for j in range(TN):
            m.d.comb += mac.b_row[j].eq(
                b_rp.data[j * IN_W : (j + 1) * IN_W].as_signed()
            )
        m.d.comb += mac.en.eq(en_d)

        cacc = Array(mac.c)
        fi = Signal(range(TM))
        fj = Signal(range(TN))
        fdone = Signal()
        done_cnt = Signal(range(TM * TN + 1))
        oi = Signal(range(TM))
        oj = Signal(range(TN))
        word = Signal(TM * IN_W)
        c_we_r = Signal()
        c_waddr_r = Signal(range(self.c_depth))
        m.d.comb += [
            c_wp.en.eq(c_we_r),
            c_wp.addr.eq(c_waddr_r),
            c_wp.data.eq(word),
            rq.acc.eq(cacc[fi * TN + fj]),
        ]
        m.d.sync += [mac.clear.eq(0), en_d.eq(0), c_we_r.eq(0)]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.done.eq(1)
                with m.If(self.start):
                    m.d.sync += [mi.eq(0), nj.eq(0)]
                    m.next = "CLEAR"
            with m.State("CLEAR"):
                m.d.comb += self.busy.eq(1)
                m.d.sync += [mac.clear.eq(1), req_k.eq(0)]
                m.next = "ACC"
            with m.State("ACC"):
                m.d.comb += self.busy.eq(1)
                issuing = req_k < K
                with m.If(issuing):
                    m.d.sync += [en_d.eq(1), req_k.eq(req_k + 1)]
                with m.Elif(~en_d):
                    m.d.sync += [
                        fi.eq(0),
                        fj.eq(0),
                        fdone.eq(0),
                        done_cnt.eq(0),
                        oi.eq(0),
                        oj.eq(0),
                    ]
                    m.next = "EVAC"
            with m.State("EVAC"):
                m.d.comb += self.busy.eq(1)
                m.d.comb += rq.in_valid.eq(~fdone)
                with m.If(~fdone):
                    with m.If(fi == TM - 1):
                        m.d.sync += fi.eq(0)
                        with m.If(fj == TN - 1):
                            m.d.sync += fdone.eq(1)
                        with m.Else():
                            m.d.sync += fj.eq(fj + 1)
                    with m.Else():
                        m.d.sync += fi.eq(fi + 1)
                with m.If(rq.out_valid):
                    m.d.sync += word.word_select(oi, IN_W).eq(rq.out)
                    m.d.sync += done_cnt.eq(done_cnt + 1)
                    with m.If(oi == TM - 1):
                        m.d.sync += [
                            c_we_r.eq(1),
                            c_waddr_r.eq(mi * self.N + nj * TN + oj),
                            oi.eq(0),
                            oj.eq(oj + 1),
                        ]
                    with m.Else():
                        m.d.sync += oi.eq(oi + 1)
                with m.If(done_cnt == TM * TN):
                    m.next = "NEXT"
            with m.State("NEXT"):
                m.d.comb += self.busy.eq(1)
                with m.If(nj == self.NTN - 1):
                    m.d.sync += nj.eq(0)
                    with m.If(mi == self.MTM - 1):
                        m.next = "IDLE"
                    with m.Else():
                        m.d.sync += mi.eq(mi + 1)
                        m.next = "CLEAR"
                with m.Else():
                    m.d.sync += nj.eq(nj + 1)
                    m.next = "CLEAR"
        return m


# ── full FFN: requant2(GELU(requant1(x.W1)).W2) ───────────────────────────────


class FFN(Elaboratable):
    """Two MatMulEngines + a parallel GELU streaming-copy between them."""

    def __init__(self, m_dim, k1, k2, n_dim, tile_m, tile_n):
        self.M, self.K1, self.K2, self.N = m_dim, k1, k2, n_dim
        self.tile_m, self.tile_n = tile_m, tile_n
        self.eng1 = MatMulEngine(m_dim, k1, k2, tile_m, tile_n)
        self.eng2 = MatMulEngine(m_dim, k2, n_dim, tile_m, tile_n)
        self._hidden_depth = self.eng1.c_depth

        self.xa_we = Signal()
        self.xa_waddr = Signal(range(self.eng1.a_depth))
        self.xa_wdata = Signal(tile_m * IN_W)
        self.w1_we = Signal()
        self.w1_waddr = Signal(range(self.eng1.b_depth))
        self.w1_wdata = Signal(tile_n * IN_W)
        self.w2_we = Signal()
        self.w2_waddr = Signal(range(self.eng2.b_depth))
        self.w2_wdata = Signal(tile_n * IN_W)
        self.g_load_en = Signal()
        self.g_load_addr = Signal(IN_W)
        self.g_load_data = Signal(IN_W)
        self.mult1 = Signal(32)
        self.shift1 = Signal(6)
        self.mult2 = Signal(32)
        self.shift2 = Signal(6)
        self.start = Signal()
        self.busy = Signal()
        self.done = Signal()
        self.c_raddr = Signal(range(self.eng2.c_depth))
        self.c_rdata = Signal(tile_m * IN_W)

    def elaborate(self, platform):
        m = Module()
        TM = self.tile_m
        m.submodules.eng1 = eng1 = self.eng1
        m.submodules.eng2 = eng2 = self.eng2
        gelus = [Gelu() for _ in range(TM)]
        for i, g in enumerate(gelus):
            m.submodules[f"gelu{i}"] = g

        m.d.comb += [
            eng1.a_we.eq(self.xa_we),
            eng1.a_waddr.eq(self.xa_waddr),
            eng1.a_wdata.eq(self.xa_wdata),
            eng1.b_we.eq(self.w1_we),
            eng1.b_waddr.eq(self.w1_waddr),
            eng1.b_wdata.eq(self.w1_wdata),
            eng2.b_we.eq(self.w2_we),
            eng2.b_waddr.eq(self.w2_waddr),
            eng2.b_wdata.eq(self.w2_wdata),
            eng1.mult.eq(self.mult1),
            eng1.shift.eq(self.shift1),
            eng2.mult.eq(self.mult2),
            eng2.shift.eq(self.shift2),
            eng2.c_raddr.eq(self.c_raddr),
            self.c_rdata.eq(eng2.c_rdata),
        ]
        for g in gelus:
            m.d.comb += [
                g.load_en.eq(self.g_load_en),
                g.load_addr.eq(self.g_load_addr),
                g.load_data.eq(self.g_load_data),
            ]

        D = self._hidden_depth
        rcnt = Signal(range(D + 1))
        wcnt = Signal(range(D + 1))
        s1_valid = Signal()
        s1_addr = Signal(range(D))
        s2_valid = Signal()
        s2_addr = Signal(range(D))
        for i, g in enumerate(gelus):
            m.d.comb += [
                g.in_code.eq(eng1.c_rdata[i * IN_W : (i + 1) * IN_W].as_signed()),
                g.in_valid.eq(s1_valid),
            ]
        gel_word = Cat(*[g.out for g in gelus])
        m.d.comb += [
            eng2.a_wdata.eq(gel_word),
            eng2.a_waddr.eq(s2_addr),
            eng2.a_we.eq(s2_valid & gelus[0].out_valid),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.done.eq(1)
                with m.If(self.start):
                    m.next = "RUN1"
            with m.State("RUN1"):
                m.d.comb += [self.busy.eq(1), eng1.start.eq(1)]
                m.next = "WAIT1"
            with m.State("WAIT1"):
                m.d.comb += self.busy.eq(1)
                with m.If(eng1.done):
                    m.d.sync += [rcnt.eq(0), wcnt.eq(0), s1_valid.eq(0), s2_valid.eq(0)]
                    m.next = "GELU"
            with m.State("GELU"):
                m.d.comb += self.busy.eq(1)
                reading = rcnt < D
                m.d.comb += eng1.c_raddr.eq(rcnt)
                with m.If(reading):
                    m.d.sync += rcnt.eq(rcnt + 1)
                m.d.sync += [
                    s1_valid.eq(reading),
                    s1_addr.eq(rcnt),
                    s2_valid.eq(s1_valid),
                    s2_addr.eq(s1_addr),
                ]
                with m.If(s2_valid & gelus[0].out_valid):
                    m.d.sync += wcnt.eq(wcnt + 1)
                with m.If(wcnt == D):
                    m.next = "RUN2"
            with m.State("RUN2"):
                m.d.comb += [self.busy.eq(1), eng2.start.eq(1)]
                m.next = "WAIT2"
            with m.State("WAIT2"):
                m.d.comb += self.busy.eq(1)
                with m.If(eng2.done):
                    m.next = "IDLE"
        return m


# ── the user_design: Wishbone B4 MMIO slave ───────────────────────────────────

# Baked-in FFN shape (the client mirrors these). 4x hidden expansion (8->32).
M, K1, K2, N = 4, 8, 32, 8
TILE = 4

CTRL, TARGET, PTR = 0, 1, 2
MULT1, SHIFT1, MULT2, SHIFT2 = 3, 4, 5, 6
ID_M, ID_K1, ID_K2, ID_N, ID_TILE = 7, 8, 9, 10, 11
DATA_WORD = 16
T_X, T_W1, T_W2, T_LUT = 0, 1, 2, 3


class FFNSlave(Elaboratable):
    """Wishbone B4 slave exposing the FFN engine through the SoC's MMIO window."""

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
        m.submodules.ffn = ffn = FFN(M, K1, K2, N, TILE, TILE)

        target = Signal(2)
        ptr = Signal(16)
        start_r = Signal()
        m.d.comb += ffn.start.eq(start_r)

        with m.If(self.wb_cyc & self.wb_stb & ~self.wb_ack):
            m.d.sync += self.wb_ack.eq(1)
        with m.Else():
            m.d.sync += self.wb_ack.eq(0)
        with m.If(start_r):
            m.d.sync += start_r.eq(0)

        active = self.wb_cyc & self.wb_stb & ~self.wb_ack
        wr = active & self.wb_we
        in_data = self.wb_adr >= DATA_WORD
        off = (self.wb_adr - DATA_WORD)[:16]
        buf_addr = (ptr + off)[:16]

        with m.If(wr & ~in_data):
            with m.Switch(self.wb_adr):
                with m.Case(CTRL):
                    m.d.sync += start_r.eq(self.wb_dat_w[0])
                with m.Case(TARGET):
                    m.d.sync += target.eq(self.wb_dat_w[:2])
                with m.Case(PTR):
                    m.d.sync += ptr.eq(self.wb_dat_w[:16])
                with m.Case(MULT1):
                    m.d.sync += ffn.mult1.eq(self.wb_dat_w)
                with m.Case(SHIFT1):
                    m.d.sync += ffn.shift1.eq(self.wb_dat_w[:6])
                with m.Case(MULT2):
                    m.d.sync += ffn.mult2.eq(self.wb_dat_w)
                with m.Case(SHIFT2):
                    m.d.sync += ffn.shift2.eq(self.wb_dat_w[:6])

        data_wr = wr & in_data
        m.d.comb += [
            ffn.xa_waddr.eq(buf_addr),
            ffn.xa_wdata.eq(self.wb_dat_w[: TILE * IN_W]),
            ffn.w1_waddr.eq(buf_addr),
            ffn.w1_wdata.eq(self.wb_dat_w[: TILE * IN_W]),
            ffn.w2_waddr.eq(buf_addr),
            ffn.w2_wdata.eq(self.wb_dat_w[: TILE * IN_W]),
            ffn.g_load_addr.eq(buf_addr[:IN_W]),
            ffn.g_load_data.eq(self.wb_dat_w[:IN_W]),
            ffn.xa_we.eq(data_wr & (target == T_X)),
            ffn.w1_we.eq(data_wr & (target == T_W1)),
            ffn.w2_we.eq(data_wr & (target == T_W2)),
            ffn.g_load_en.eq(data_wr & (target == T_LUT)),
            ffn.c_raddr.eq(Mux(in_data, buf_addr, 0)),
        ]

        with m.Switch(self.wb_adr):
            with m.Case(CTRL):
                m.d.comb += self.wb_dat_r.eq(Cat(ffn.done, ffn.busy))
            with m.Case(TARGET):
                m.d.comb += self.wb_dat_r.eq(target)
            with m.Case(PTR):
                m.d.comb += self.wb_dat_r.eq(ptr)
            with m.Case(ID_M):
                m.d.comb += self.wb_dat_r.eq(M)
            with m.Case(ID_K1):
                m.d.comb += self.wb_dat_r.eq(K1)
            with m.Case(ID_K2):
                m.d.comb += self.wb_dat_r.eq(K2)
            with m.Case(ID_N):
                m.d.comb += self.wb_dat_r.eq(N)
            with m.Case(ID_TILE):
                m.d.comb += self.wb_dat_r.eq(TILE)
            with m.Default():
                m.d.comb += self.wb_dat_r.eq(ffn.c_rdata)

        return m
