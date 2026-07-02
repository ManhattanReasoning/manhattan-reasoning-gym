"""Drive the example MAC design on a real cloud FPGA.

Run with:

    mrg run examples/app.py

``mrg run`` imports this file, programs the FPGA with ``design`` (first run
~2-3 min), then calls the ``@app.local_entrypoint()`` function. The board
stays reserved to you after the entrypoint returns — keep iterating with
``mrg run examples/app.py --no-program`` / ``mrg read`` / ``mrg write``, and
free it with ``mrg reset <fpga_id>`` (or ``app.release()``) when done.

Needs an API key: ``mrg login``, ``$MRG_API_KEY``, or ``--api-key``.
"""

from pathlib import Path

import manhattan_reasoning_gym as mrg

DESIGN = Path(__file__).with_name("design.py")


class MacRegs(mrg.RegisterMap):
    """design.py's Wishbone map. The bus is byte-addressed: word n => 4*n."""

    A          = 0x0  # write: set operand A
    ACCUMULATE = 0x4  # write: acc += A * value
    ACC        = 0x8  # read:  accumulator, low 32 bits
    RESET      = 0xC  # write: clear the accumulator


app = mrg.cloud.App(
    "mac_example",
    design=str(DESIGN),
    registers=MacRegs,
)


@app.local_entrypoint()
def main():
    app.write(MacRegs.RESET, 0)
    app.write(MacRegs.A, 3)
    app.write(MacRegs.ACCUMULATE, 4)  # acc += 3 * 4
    app.write(MacRegs.ACCUMULATE, 5)  # acc += 3 * 5
    result = app.read(MacRegs.ACC)
    print(f"3*4 + 3*5 = {result}")
    assert result == 27, f"expected 27, got {result}"
