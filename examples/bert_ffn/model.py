"""A small but architecturally faithful BERT encoder block (NumPy).

This is a real transformer block -- token + positional embeddings, multi-head
self-attention, a position-wise feed-forward network with the BERT 4x hidden
expansion and GELU, residual connections, and LayerNorm. Dimensions are kept
tiny (``d_model=8``, ``d_ff=32``) for one reason: the **feed-forward sublayer
runs end-to-end on the Cloud FPGA's INT8 FFN accelerator**
(``examples/ffn_accel``), whose hardware shape is fixed at 8->32->8.

Everything except the FFN runs in float NumPy on the host. The FFN -- both
projections, the int8 requantization, and GELU -- executes on-chip in a single
hardware call, injected as ``ffn_fn(x, W1, W2) -> out``. Weights are random but
seeded, so runs are reproducible; the point of the demo is the *architecture and
the FPGA data path*, not a trained checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# The FFN backend: takes activations x [seq,d] and the two projection weight
# matrices, returns the FFN sublayer output [seq,d]. In the live demo this is
# the FPGA accelerator (accel.run_ffn); the --sim path uses accel.sim_ffn.
FFNBackend = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Config:
    vocab_size: int
    d_model: int = 8
    n_heads: int = 2
    d_ff: int = 32
    max_seq: int = 8
    seed: int = 0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ── math helpers ────────────────────────────────────────────────────────────--


def gelu(x: np.ndarray) -> np.ndarray:
    """BERT's tanh-approximation GELU."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def layernorm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ── the encoder block ─────────────────────────────────────────────────────────


class BertBlock:
    """One BERT encoder layer with randomly-initialized, seeded weights."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d, ff, vocab = cfg.d_model, cfg.d_ff, cfg.vocab_size

        def w(*shape: int) -> np.ndarray:
            # Small init so activations stay in a well-conditioned range.
            return rng.standard_normal(shape) * (1.0 / np.sqrt(shape[0]))

        self.tok_emb = w(vocab, d)
        self.pos_emb = w(cfg.max_seq, d)

        # Self-attention projections.
        self.Wq, self.Wk, self.Wv, self.Wo = (w(d, d) for _ in range(4))
        self.ln1_g, self.ln1_b = np.ones(d), np.zeros(d)

        # Feed-forward: d -> ff -> d (this whole sublayer runs on the FPGA).
        self.W1, self.W2 = w(d, ff), w(ff, d)
        self.ln2_g, self.ln2_b = np.ones(d), np.zeros(d)

    # -- stages -------------------------------------------------------------

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        seq = len(token_ids)
        return self.tok_emb[token_ids] + self.pos_emb[:seq]

    def self_attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention, in float NumPy (not the FPGA path)."""
        cfg = self.cfg
        seq = x.shape[0]
        q = (x @ self.Wq).reshape(seq, cfg.n_heads, cfg.d_head)
        k = (x @ self.Wk).reshape(seq, cfg.n_heads, cfg.d_head)
        v = (x @ self.Wv).reshape(seq, cfg.n_heads, cfg.d_head)
        out = np.empty((seq, cfg.n_heads, cfg.d_head))
        for h in range(cfg.n_heads):
            scores = q[:, h] @ k[:, h].T / np.sqrt(cfg.d_head)
            out[:, h] = softmax(scores) @ v[:, h]
        return out.reshape(seq, cfg.d_model) @ self.Wo

    def feed_forward(self, x: np.ndarray, ffn_fn: FFNBackend) -> np.ndarray:
        """The BERT FFN ``GELU(x.W1).W2`` -- run entirely on the FPGA accelerator."""
        return ffn_fn(x, self.W1, self.W2)

    # -- full forward -------------------------------------------------------

    def forward(self, token_ids: np.ndarray, ffn_fn: FFNBackend) -> np.ndarray:
        """Run the block end to end; the FFN sublayer uses ``ffn_fn`` (the FPGA)."""
        x = self.embed(token_ids)
        x = layernorm(x + self.self_attention(x), self.ln1_g, self.ln1_b)
        x = layernorm(x + self.feed_forward(x, ffn_fn), self.ln2_g, self.ln2_b)
        return x

    # -- exact float references (no quantization, no FPGA) ------------------

    def feed_forward_float(self, x: np.ndarray) -> np.ndarray:
        """The FFN in exact float64 -- baseline for quantization error."""
        return gelu(x @ self.W1) @ self.W2

    def forward_float(self, token_ids: np.ndarray) -> np.ndarray:
        """Full block in exact float64, for an end-to-end reference."""
        x = self.embed(token_ids)
        x = layernorm(x + self.self_attention(x), self.ln1_g, self.ln1_b)
        x = layernorm(x + self.feed_forward_float(x), self.ln2_g, self.ln2_b)
        return x
