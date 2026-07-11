"""Run the INT8 FFN accelerator on a live Cloud FPGA via the SDK.

    mrg run examples/ffn_accel/client_sdk.py

Builds a quantization plan host-side (numpy), streams the int8 operands, requant
constants, and GELU table into the board through the MMIO window, runs the whole
GELU(x.W1).W2 pipeline in hardware, and checks the int8 result is **bit-exact**
against the golden integer model -- then prints the dequantized output next to
the float reference. The MMIO driver lives in `accel.py`. Run `mrg login` first.
"""

import accel  # shared host-side driver (register map + load/run/verify)
import golden
import manhattan_reasoning_gym as mrg
import numpy as np

app = mrg.cloud.App(
    "ffn_accel", design="examples/ffn_accel/design.py", registers=accel.Regs
)

# FPGA INT8 TRANSFORMER FEED-FORWARD ACCELERATOR
#   requant2( GELU( requant1(x . W1) ) . W2 )  -- tiling, requant, GELU all on-chip.


@app.local_entrypoint()
def main():
    with app:
        accel.check_dims(app)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((accel.M, accel.K1))
        W1 = rng.standard_normal((accel.K1, accel.K2)) / np.sqrt(accel.K1)
        W2 = rng.standard_normal((accel.K2, accel.N)) / np.sqrt(accel.K2)

        print("running FFN on silicon ...")
        deq = accel.run_ffn(app, x, W1, W2)  # verifies bit-exact internally
        print("  bit-exact against golden int8 model  ✅")

        ref = golden.gelu(x @ W1) @ W2  # exact float FFN
        rel = np.abs(deq - ref).max() / (np.abs(ref).max() + 1e-9)
        print(f"max relative error vs float FFN: {rel:.4f}")
        print("FPGA output (dequantized):")
        for row in deq:
            print("  " + "  ".join(f"{v:+.3f}" for v in row))
