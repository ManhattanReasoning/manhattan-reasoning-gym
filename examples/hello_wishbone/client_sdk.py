"""hello_wishbone example using the Cloud FPGA SDK.

Run with:
    mrg run examples/hello_wishbone/client_sdk.py

The SDK programs the FPGA automatically, then calls main().
Run `mrg login` first (or set MRG_API_KEY).
"""

import manhattan_reasoning_gym as mrg


class Regs(mrg.cloud.RegisterMap):
    # The EchoSlave exposes a 512-word (2 KB) echo RAM starting at byte 0.
    ECHO = 0x0000


app = mrg.cloud.App(
    "hello_wishbone",
    design="examples/hello_wishbone/design.py",
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
