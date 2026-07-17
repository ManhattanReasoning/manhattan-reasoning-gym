"""verilog_hello example using the Cloud FPGA SDK -- plain Verilog, not Amaranth.

Run with:
    mrg run examples/verilog_hello/client_sdk.py

The SDK programs the FPGA automatically, then calls main(). design.v's top
module (echo_slave) is auto-detected, same as an Amaranth design -- pass
top="..." to App() only if a file has more than one module exposing the
Wishbone contract. Run `mrg login` first (or set MRG_API_KEY).
"""

import manhattan_reasoning_gym as mrg


class Regs(mrg.cloud.RegisterMap):
    # echo_slave exposes a 512-word (2 KB) echo RAM starting at byte 0.
    ECHO = 0x0000


app = mrg.cloud.App(
    "verilog_hello",
    design="examples/verilog_hello/design.v",
    registers=Regs,
)


@app.local_entrypoint()
def main():
    with app:
        pattern = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xABCDEF01]

        print("writing pattern ...")
        for i, word in enumerate(pattern):
            app.write(Regs.ECHO + i * 4, word)

        print("reading back ...")
        for i, expected in enumerate(pattern):
            got = app.read(Regs.ECHO + i * 4)
            status = "OK" if got == expected else f"MISMATCH (got {got:#010x})"
            print(f"  [{i}] {expected:#010x} → {status}")
