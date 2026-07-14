"""Minimal app.stream() smoke test.

Debug harness for isolating a "stream write ... failed" error seen on a more
complex design: EchoSlave is a trivial 512-word memory (see design.py) that
builds in well under a minute, so this exercises the SDK -> orchestrator ->
nginx -> host agent -> firmware -> Wishbone path in isolation from any
slower design's own build time or RTL behavior.

A fresh build no longer pins a board -- the server assigns whichever one
frees up first once the build finishes, and ``app.fpga_id`` reports which
once ``with app:`` has programmed it. To retest a *specific* board you
already have a live session on, skip programming and reconnect:
``mrg run examples/hello_wishbone/stream_client_sdk.py --no-program --fpga-id N``.

Run with:
    mrg run examples/hello_wishbone/stream_client_sdk.py

Exercises, in order:
  1. A single write()/read() over the stream (baseline sanity).
  2. A normal (incrementing-address) burst write + read-back.
  3. A fixed_address=True burst write (same pattern SILICA's LITERAL_IN
     push-register load uses) + read-back, to confirm the fixed-address
     write path itself works on known-good hardware.
"""

import manhattan_reasoning_gym as mrg


class Regs(mrg.cloud.RegisterMap):
    ECHO = 0x0000


app = mrg.cloud.App(
    "hello_wishbone_stream_smoke_test",
    design="examples/hello_wishbone/design.py",
    registers=Regs,
)


@app.local_entrypoint()
def main():
    with app:
        print(f"landed on fpga{app.fpga_id}")
        with app.stream() as s:
            print("-- single write/read --")
            s.write(Regs.ECHO, 0xDEADBEEF)
            got = s.read(Regs.ECHO)
            ok = "OK" if got == 0xDEADBEEF else "MISMATCH"
            print(f"  wrote 0xdeadbeef, read back {got:#010x} -> {ok}")

            print("-- incrementing burst write/read --")
            pattern = [0x1111, 0x2222, 0x3333, 0x4444]
            s.write(Regs.ECHO, pattern)
            got_burst = s.read(Regs.ECHO, count=len(pattern))
            ok = "OK" if got_burst == pattern else "MISMATCH"
            print(f"  wrote {[hex(w) for w in pattern]}")
            print(f"  read  {[hex(w) for w in got_burst]} -> {ok}")

            print("-- fixed_address burst write/read (LITERAL_IN-style) --")
            pushed = [0xAAAA0000 | i for i in range(8)]
            s.write(Regs.ECHO, pushed, fixed_address=True)
            got_fixed = s.read(Regs.ECHO)
            ok = "OK (last word won)" if got_fixed == pushed[-1] else "MISMATCH"
            print(
                f"  pushed {len(pushed)} words to a fixed address, "
                f"last write was {pushed[-1]:#010x}"
            )
            print(f"  read back {got_fixed:#010x} -> {ok}")

        print("stream smoke test passed")
