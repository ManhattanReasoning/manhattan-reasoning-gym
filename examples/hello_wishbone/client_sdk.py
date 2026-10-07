"""hello_wishbone example using the Cloud FPGA SDK.

Run with:
    mrg run examples/hello_wishbone/client_sdk.py

The SDK programs the FPGA automatically, then calls main(). Every read/write
goes over one WebSocket the SDK manages; on a network that blocks WebSockets,
run it with MRG_TRANSPORT=http (or --transport http) instead.
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

        print("burst write + read-back ...")
        burst = [0x1111, 0x2222, 0x3333, 0x4444]
        app.write(Regs.ECHO, burst)
        got_burst = app.read(Regs.ECHO, count=len(burst))
        status = "OK" if got_burst == burst else "MISMATCH"
        print(f"  wrote {[hex(w) for w in burst]}")
        print(f"  read  {[hex(w) for w in got_burst]} → {status}")

        # fixed_address=True repeats the address for every word (a FIFO /
        # push-register port); on this plain memory the last word wins.
        print("fixed-address burst write ...")
        pushed = [0xAAAA0000 | i for i in range(8)]
        app.write(Regs.ECHO, pushed, fixed_address=True)
        got_fixed = app.read(Regs.ECHO)
        status = "OK (last word won)" if got_fixed == pushed[-1] else "MISMATCH"
        print(f"  pushed {len(pushed)} words, read back {got_fixed:#010x} → {status}")
