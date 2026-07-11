"""BERT transformer pipeline with the feed-forward layer on the Cloud FPGA.

A real (small) BERT encoder block runs over a tokenized sentence: WordPiece
tokenizer -> embeddings -> multi-head self-attention -> **feed-forward network
that runs end-to-end on the FPGA** -> residual + LayerNorm. The FFN sublayer
(both projections + int8 requantization + GELU) executes on the INT8 FFN
accelerator in ``examples/ffn_accel`` -- one hardware call, not host-side tiling.

Run on hardware (programs an idle FPGA, then runs the pipeline):

    mrg run examples/bert_ffn/client_sdk.py

Hardware-free smoke test (no FPGA, golden backend -- bit-identical to the board):

    python examples/bert_ffn/client_sdk.py --sim --text "fpga runs bert"

If a run ever leaves the FPGA stuck in ``error``, recover it with
``mrg reset <fpga_id>``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# The FFN accelerator (design + host driver + golden model) lives in the sibling
# ffn_accel example; make it importable and reuse it unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ffn_accel"))

import accel  # noqa: E402  ffn_accel host-side MMIO driver
from model import BertBlock, Config  # noqa: E402

import manhattan_reasoning_gym as mrg  # noqa: E402

# Reuse the ffn_accel design + its register map. A BERT FFN sublayer is exactly
# what this accelerator computes; dimensions are pinned to its 8->32->8 shape.
app = mrg.cloud.App(
    "bert_ffn", design="examples/ffn_accel/design.py", registers=accel.Regs
)

DEFAULT_TEXT = "fpga runs bert"


# ── tokenizer ─────────────────────────────────────────────────────────────────


def tokenize(text: str, max_seq: int) -> tuple[np.ndarray, list[str], int]:
    """WordPiece-tokenize ``text`` to <= max_seq ids; returns ids, tokens, vocab."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ids = tok(text)["input_ids"][:max_seq]
    return np.array(ids, dtype=np.int64), tok.convert_ids_to_tokens(ids), tok.vocab_size


def _show_matrix(name: str, mat: np.ndarray) -> None:
    print(f"  {name}  (shape {mat.shape[0]}x{mat.shape[1]}):")
    for row in mat:
        print("    " + "  ".join(f"{v:+7.3f}" for v in row))


# ── the pipeline ──────────────────────────────────────────────────────────────


def run_pipeline(text: str, *, use_fpga: bool) -> None:
    """Tokenize, build the block, run the FFN on FPGA (or golden), report fidelity."""
    # Dims are pinned to the accelerator's fixed 8->32->8 shape (BERT's 4x FFN
    # expansion); seq <= the hardware's M (zero-padded inside the driver).
    cfg = Config(vocab_size=0, d_model=accel.K1, n_heads=2, d_ff=accel.K2,
                 max_seq=accel.M, seed=0)

    token_ids, tokens, vocab = tokenize(text, cfg.max_seq)
    cfg.vocab_size = vocab
    block = BertBlock(cfg)

    print(f"\n  input    : {text!r}")
    print(f"  tokens   : {tokens}")
    print(f"  token ids: {token_ids.tolist()}")
    print(
        f"  config   : d_model={cfg.d_model}  d_ff={cfg.d_ff}  "
        f"heads={cfg.n_heads}  seq={len(token_ids)}\n"
    )

    # Pick the FFN backend: the live accelerator, or the bit-identical golden sim.
    if use_fpga:
        accel.check_dims(app)
        ffn_fn = lambda x, W1, W2: accel.run_ffn(app, x, W1, W2)  # noqa: E731
        backend_name = f"FPGA {app.name} (fpga{app.fpga_id})"
    else:
        ffn_fn = accel.sim_ffn
        backend_name = "golden sim (no hardware)"

    print(f"  feed-forward backend: {backend_name}")
    print("  running the FFN sublayer on the accelerator ...")
    t0 = time.time()
    hidden = block.forward(token_ids, ffn_fn)
    dt = time.time() - t0
    print(f"  done in {dt:.2f}s\n")

    # References: the same block with the golden FFN (bit-identical to hardware)
    # and an exact-float forward (shows the int8 quantization error).
    hidden_hostint = block.forward(token_ids, accel.sim_ffn)
    hidden_float = block.forward_float(token_ids)

    hw_vs_host = float(np.abs(hidden - hidden_hostint).max())
    rel_vs_float = float(
        np.linalg.norm(hidden - hidden_float) / (np.linalg.norm(hidden_float) + 1e-12)
    )

    _show_matrix("block output (FPGA FFN path)", hidden)
    print()
    print(f"  FPGA vs golden FFN  : max abs diff = {hw_vs_host:.3e} "
          f"({'EXACT MATCH' if hw_vs_host == 0 else 'MISMATCH'})")
    print(f"  int8 FFN vs float   : relative L2 error = {rel_vs_float:.3%}")

    cls = hidden[0]
    print("\n  [CLS] sentence embedding: "
          + "  ".join(f"{v:+.3f}" for v in cls))

    if use_fpga and hw_vs_host != 0:
        raise AssertionError(
            f"FPGA FFN disagreed with the golden reference (max diff {hw_vs_host})"
        )


# ── entrypoints ───────────────────────────────────────────────────────────────


def _run_on_hardware(text: str) -> None:
    with app:
        run_pipeline(text, use_fpga=True)


@app.local_entrypoint()
def main() -> None:
    """Real run: program an idle FPGA (done by the CLI), run the FFN on hardware."""
    _run_on_hardware(DEFAULT_TEXT)


if __name__ == "__main__":
    # Direct `python client_sdk.py` defaults to --sim (safe offline smoke test).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="input sentence")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--sim", action="store_true", help="golden backend (default)")
    grp.add_argument("--hw", action="store_true", help="drive the live FPGA")
    args = parser.parse_args()

    if args.hw:
        app._program()
        _run_on_hardware(args.text)
    else:
        run_pipeline(args.text, use_fpga=False)
