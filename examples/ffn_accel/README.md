# ffn_accel — a real INT8 transformer feed-forward accelerator

A **streaming, tiled, quantized FFN engine** in Amaranth that runs the BERT
feed-forward sublayer — `requant2(GELU(requant1(x·W1))·W2)` — entirely in
silicon. Unlike [`../bert_ffn`](../bert_ffn), which used the host to tile a 4×4
toy core, **this engine owns the whole computation**: it tiles, accumulates over
K, requantizes, and applies GELU on-chip. It builds and runs on the live cloud
board through the existing 2 KB Wishbone MMIO window — no SoC changes.

```
 x (int8) ─▶ MatMul-1 ─▶ Requant ─▶ GELU LUT ─▶ MatMul-2 ─▶ Requant ─▶ y (int8)
             x·W1        int32→int8   int8→int8   ·W2         int32→int8
             └─ tiled output-stationary MAC grid (tile×tile INT8 MACs) ─┘
```

## What's "real" about it

| Feature | How (all in `design.py`) |
|---|---|
| **Parallel MAC grid** | `tile×tile` output-stationary INT8→INT32 MACs, one DSP each (`MacArray`) |
| **Hardware tiling** | FSM sweeps M×N output tiles, accumulates over K from packed BRAM (`MatMulEngine`) |
| **Proper quantization** | per-tensor int8 + gemmlowp-style fixed-point requant: `(acc·mult + bias) >> shift` (`Requant`) |
| **GELU** | exact int8→int8 lookup ROM, runtime-loadable per model (`Gelu`) |
| **Native chaining** | each engine emits results in the layout the next consumes — GELU is a 1-pass streaming copy, no transpose (`FFN`) |
| **Pipelined** | address-ahead-of-data K-loop (maps to sync BRAM), 1-cycle requant/GELU stages |

`design.py` is one self-contained file (the orchestrator uploads only that), and
everything in it is **bit-exact** against a NumPy golden model (`golden.py`) —
the golden model performs the identical integer ops, so a passing sim test *is*
a proof of correctness, not an approximation. The host-side MMIO driver
(`accel.py`) is shared with `../bert_ffn`, which uses this accelerator for its
entire feed-forward sublayer.

## ECP5-85 fit (real `yosys synth_ecp5`)

The whole dual-engine slave (`design.py`) at `tile=4`:

| | LUT4 | carry (CCU2C) | DSP (MULT18X18) | BRAM (DP16KD) | FF |
|---|---|---|---|---|---|
| FFNSlave | 3,437 | 840 | 40 | 6 | 1,505 |
| **Budget (LFE5UM5G-85F)** | 84k | — | **156** | 208 | — |

Fits with huge headroom next to the VexRiscv+LiteEth SoC. The parallel MAC grid
scales as `tile²` DSPs — **12×12 (144 MACs) is the max all-DSP fit**; 16×16
(256 MACs) needs ~100 LUT-fabric multipliers. See `python synth.py --help`.

## Run it on the cluster

```sh
mrg login                                   # or set MRG_API_KEY
mrg run examples/ffn_accel/client_sdk.py    # builds the bitstream, programs a board, runs
```

The client builds a quantization plan host-side, streams the int8 operands +
requant constants + GELU table through the MMIO window, runs the pipeline, reads
the int8 output back, and asserts it is bit-exact against the golden model.

## Tests

```sh
cd examples/ffn_accel && python -m pytest tests/ -q   # 9 sim tests, no hardware
```

Covers the MAC grid, requant, the tiling engine, the full FFN, and the complete
Wishbone MMIO data path — each bit-exact against `golden.py`.

## The honest limitation (why dims are small)

A *full* BERT-base FFN has 4.7 MB of weights — **10× the ECP5-85's 468 KB of
BRAM** — so a real deployment must *stream* weights from off-chip, and on this
board the only host link is 100 Mbit Ethernet through a softcore, which caps a
full layer at ~0.5 s of weight transfer (it's I/O-bound, not compute-bound). The
engine here is architected for that streaming model (operands flow in tile by
tile); the on-board demo keeps dimensions small (`8→32→8`) because the operands
load through the 2 KB MMIO window one word at a time. The *silicon* is the real
thing; feeding it at full BERT scale is a data-path project (DMA + faster link),
not a compute one.
