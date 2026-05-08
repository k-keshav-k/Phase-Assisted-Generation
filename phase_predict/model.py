"""Transformer-based model for phase-tuple sequence prediction.

# Model choice rationale
# -----------------------
# We need to predict the next integer tuple (block_size, refinement_steps)
# refinement_steps) given the previous n such tuples.  Several architectures
# were considered:
#
#   Variable-order Markov (baseline)
#       Only captures co-occurrence statistics within a fixed order.
#       Cannot generalise to unseen tuple patterns and has no internal
#       representation of temporal structure.
#
#   LSTM / GRU (recurrent)
#       Sequential computation means O(n) forward passes that cannot be
#       parallelised during training.  Known to struggle with long-range
#       dependencies despite gating, and require careful gradient clipping.
#
#   TCN (Temporal Convolutional Network)
#       Strong for very long, fixed-period sequences but less flexible for
#       variable-length contexts.  The convolutional receptive field grows
#       only with depth/dilation, making architecture selection less
#       transparent.
#
#   Transformer encoder (chosen)
#       Self-attention considers all O(n²) pairwise relationships between
#       context positions in a single layer, capturing long-range structure.
#       All positions are processed in parallel, making training fast.
#       Positional encoding handles ordering explicitly.
#       The regression output head is trivially swappable for classification
#       heads if the integer vocabulary is small and known.
#       A 2-layer, 4-head, d_model=64 encoder is compact and fast on CPU,
#       yet still GPU-scalable.
#
# Architecture
# ------------
#   Input  : (batch, window_size, tuple_size)  – float tensor of past tuples
#   Embed  : Linear(tuple_size → d_model)  +  sinusoidal positional encoding
#   Encode : N × TransformerEncoderLayer(d_model, n_heads, dim_feedforward)
#   Pool   : last-position token (the most recent context step)
#   Output : Linear(d_model → num_stab_thresholds)  – ordinal logits for
#            stabilising steps
#
# Targets are ordinal binary vectors and loss is BCE (one per threshold).
# Predictions are decoded by counting thresholds with sigmoid > 0.5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from phase_predict.schema import ModelConfig


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x.

        Args:
            x: float tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of same shape with positional information added.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


class PhaseTransformer(nn.Module):
    """Transformer encoder with ordinal regression head.

    Given a window of ``window_size`` past phase tuples the model produces
    ``stab_logits``: ordinal logits over ``num_stab_thresholds`` for the
    max stabilizing step of the next tuple.

    Args:
        config: :class:`~phase_predict.schema.ModelConfig` with all
            architecture hyper-parameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.input_tuple_size, config.d_model)

        self.pos_encoding = _SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_len=config.window_size + 1,
            dropout=config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.stab_head = nn.Linear(config.d_model, config.num_stab_thresholds)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.input_projection(x)
        emb = self.pos_encoding(emb)
        encoded = self.encoder(emb)
        last = encoded[:, -1, :]

        stab_logits = self.stab_head(last)
        return stab_logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply standard Transformer weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
