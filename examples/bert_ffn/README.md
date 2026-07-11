# bert_ffn

A small but **architecturally faithful BERT encoder block** whose **feed-forward
network runs on the Cloud FPGA**. The pipeline is a real transformer:

```
text → WordPiece tokenizer → embeddings → multi-head self-attention
     → FEED-FORWARD (on the FPGA) → residual + LayerNorm → block output
```

The feed-forward network is the classic BERT FFN — `GELU(x·W₁)·W₂` with the 4×
hidden expansion — and **the whole sublayer executes on the FPGA** in a single
hardware call, using the [`../ffn_accel`](../ffn_accel) INT8 accelerator: both
projections, the int8 requantization between them, and GELU all run on-chip.
Everything else (attention, LayerNorm, softmax) runs in float NumPy on the host.

## Why it's small

The accelerator's hardware shape is fixed at **8→32→8** (`d_model=8`, `d_ff=32`),
so the block is configured to match and sequences are capped at the hardware's
`M=4` (zero-padded inside the driver). This isn't a tiling limit — it's the
on-chip dimensions the bitstream was built for. Scaling up is a re-parameterize-
and-rebuild of `ffn_accel`, bounded by the board's BRAM and the MMIO load path,
not by correctness (the engine tiles internally for any size).

## How the FFN reaches the FPGA

The FFN sublayer is injected into the block as a backend `ffn_fn(x, W1, W2)`:

1. **Quantize** `x`, `W1`, `W2` to per-tensor int8 and derive the requant
   constants + GELU table (`ffn_accel/golden.build_plan`).
2. **Stream** them through the 2 KB MMIO window and run the whole pipeline
   on-chip (`ffn_accel/accel.run_ffn`); the engine tiles, accumulates, requants,
   and applies GELU itself.
3. **Read back** the int8 result, check it is **bit-exact** against the golden
   model, and **dequantize**.

The run cross-checks the hardware: the FPGA output must be **bit-exact** against
the golden FFN backend (proves the hardware is correct), and the int8 FFN is
compared to an exact-float forward to report the quantization error (a few %).

## Contents

- `client_sdk.py` — the SDK app + `@local_entrypoint`; uses
  `../ffn_accel/design.py` as the FPGA design and `../ffn_accel/accel.py` as the
  host driver. Run with `mrg run`.
- `model.py` — the BERT block in NumPy (embeddings, attention, FFN seam,
  LayerNorm). The FFN sublayer calls an injected `ffn_fn` backend.
- `tests/unit/` — pure-Python checks of the FFN backend, block forward, and the
  register map.

## Run

```sh
# Unit tests (no hardware, needs numpy)
pytest examples/bert_ffn/tests/

# Hardware-free smoke test (golden backend, real tokenizer)
python examples/bert_ffn/client_sdk.py --sim --text "fpga runs bert"

# Program an idle FPGA and run the feed-forward on hardware
#   needs:  pip install numpy transformers   and   MRG_API_KEY set
mrg run examples/bert_ffn/client_sdk.py
```

## Recovery

`with app:` releases the session on exit, reflashing the base SoC back to
`idle`. If a board ever gets stuck in `error`, recover it with
`mrg reset <fpga_id>`.

## Example output

```
  input    : 'fpga runs bert'
  tokens   : ['[CLS]', 'f', '##pg', '##a']
  config   : d_model=8  d_ff=32  heads=2  seq=4
  feed-forward backend: FPGA bert_ffn (fpga0)
  running the FFN sublayer on the accelerator ...
  ...
  FPGA vs golden FFN  : max abs diff = 0.000e+00 (EXACT MATCH)
  int8 FFN vs float   : relative L2 error = 0.83%
```
