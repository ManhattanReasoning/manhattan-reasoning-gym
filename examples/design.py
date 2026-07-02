"""A tiny, tunable example user design: a bus-driven multiply-accumulate.

Wishbone B4 slave register map. The bus is byte-addressed from software: word
address n below is byte address 4*n in app.read/app.write and mrg read/write.
    word 0 (byte 0x0)  write  set operand A
    word 1 (byte 0x4)  write  accumulate: acc += A * (value written here)
    word 2 (byte 0x8)  read   accumulator, low 32 bits
    word 3 (byte 0xC)  write  reset accumulator to 0

The signal names (wb_*) are the fixed contract the SoC binds by name — keep them
as-is. The one knob to *play with* is WIDTH below: widen the operands and watch
the DSP count and Fmax move in the synth / pnr reports.

Requires Amaranth >= 0.5.
"""

from amaranth.hdl import Elaboratable, Module, Signal

# >>> PLAY WITH ME <<<  try 8, 18, 32 and re-run the build cell / mrg.build.synth
WIDTH = 18


class MacAccumulator(Elaboratable):
    """Multiply-accumulate over the Wishbone bus (one DSP for WIDTH<=18)."""

    def __init__(self, width=WIDTH):
        self.width = width

        # Wishbone B4 slave contract (inputs from the bus master / firmware).
        self.wb_cyc = Signal()
        self.wb_stb = Signal()
        self.wb_we = Signal()
        self.wb_adr = Signal(9)  # 2 KB user region = 512 words
        self.wb_dat_w = Signal(32)
        self.wb_sel = Signal(4)
        # Wishbone outputs (driven by this slave).
        self.wb_dat_r = Signal(32)
        self.wb_ack = Signal()

    def elaborate(self, platform):
        m = Module()
        a = Signal(self.width)
        acc = Signal(48)

        active = self.wb_cyc & self.wb_stb & ~self.wb_ack
        write = active & self.wb_we

        with m.If(write):
            with m.Switch(self.wb_adr):
                with m.Case(0):
                    m.d.sync += a.eq(self.wb_dat_w[: self.width])
                with m.Case(1):
                    # The multiply infers an ECP5 DSP (MULT18X18D).
                    m.d.sync += acc.eq(acc + a * self.wb_dat_w[: self.width])
                with m.Case(3):
                    m.d.sync += acc.eq(0)

        m.d.comb += self.wb_dat_r.eq(acc[:32])

        # Registered ack: high one cycle after cyc+stb, then clears.
        with m.If(active):
            m.d.sync += self.wb_ack.eq(1)
        with m.Else():
            m.d.sync += self.wb_ack.eq(0)

        return m
