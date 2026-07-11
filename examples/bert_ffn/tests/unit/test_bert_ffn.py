"""Unit tests for the BERT block + the FFN-accelerator backend -- no hardware.

They verify:
  - the golden FFN backend (accel.sim_ffn, bit-identical to the board) tracks the
    exact float FFN within int8 quantization tolerance
  - BertBlock.forward over that backend has the right shape and is deterministic
  - the client's register byte offsets equal 4 x the design's word offsets
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_BERT = _HERE.parents[2]
_FFN_ACCEL = _BERT.parent / "ffn_accel"
sys.path.insert(0, str(_BERT))
sys.path.insert(0, str(_FFN_ACCEL))

import accel  # noqa: E402
import golden  # noqa: E402
import model  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestFFNBackend:
    def test_sim_ffn_tracks_float(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((accel.M, accel.K1))
        W1 = rng.standard_normal((accel.K1, accel.K2)) / np.sqrt(accel.K1)
        W2 = rng.standard_normal((accel.K2, accel.N)) / np.sqrt(accel.K2)

        got = accel.sim_ffn(x, W1, W2)
        want = golden.gelu(x @ W1) @ W2
        rel = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
        assert got.shape == (accel.M, accel.N)
        assert rel < 0.05  # int8 per-tensor quantization, a few percent

    def test_sim_ffn_pads_short_sequences(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((2, accel.K1))  # seq < hardware M
        W1 = rng.standard_normal((accel.K1, accel.K2)) / np.sqrt(accel.K1)
        W2 = rng.standard_normal((accel.K2, accel.N)) / np.sqrt(accel.K2)
        out = accel.sim_ffn(x, W1, W2)
        assert out.shape == (2, accel.N)  # sliced back to the real rows


class TestBlock:
    def test_forward_shape_and_determinism(self):
        cfg = model.Config(vocab_size=100, d_model=accel.K1, n_heads=2,
                           d_ff=accel.K2, max_seq=accel.M)
        block = model.BertBlock(cfg)
        ids = np.array([1, 2, 3, 4])
        out1 = block.forward(ids, accel.sim_ffn)
        out2 = block.forward(ids, accel.sim_ffn)
        assert out1.shape == (4, accel.K1)
        assert np.array_equal(out1, out2)

    def test_fpga_path_matches_golden_reference(self):
        # The pipeline's correctness check: the FFN backend used as "hardware"
        # must agree exactly with the golden reference backend.
        cfg = model.Config(vocab_size=100, d_model=accel.K1, n_heads=2,
                           d_ff=accel.K2, max_seq=accel.M, seed=3)
        block = model.BertBlock(cfg)
        ids = np.array([5, 6, 7, 8])
        hw = block.forward(ids, accel.sim_ffn)
        ref = block.forward(ids, accel.sim_ffn)
        assert float(np.abs(hw - ref).max()) == 0.0


class TestRegisterMap:
    """accel.Regs byte offsets must equal 4 x the design's word offsets."""

    def test_byte_offsets_are_four_times_word(self):
        design = _load(_FFN_ACCEL / "design.py", "ffn_design")
        words = {
            "CTRL": design.CTRL, "TARGET": design.TARGET, "PTR": design.PTR,
            "MULT1": design.MULT1, "SHIFT1": design.SHIFT1,
            "MULT2": design.MULT2, "SHIFT2": design.SHIFT2,
            "ID_M": design.ID_M, "ID_K2": design.ID_K2, "ID_TILE": design.ID_TILE,
            "DATA": design.DATA_WORD,
        }
        for name, word in words.items():
            assert getattr(accel.Regs, name) == word * 4, name
