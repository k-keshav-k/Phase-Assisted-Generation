"""Data schemas for the phase_predict package.

All public data structures are defined here to keep the rest of the package
decoupled from each other and easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


class PhaseTuple(NamedTuple):
    """One generation-block's adaptive parameters.

    Fields map directly to the scheduling decisions produced by the PAG
    scheduler:
      - block_size:         number of tokens decoded together in this block
      - stabilizing_steps:  diffusion steps at which the block's tokens first
                            reached their final identity (integer, >= 0)
      - refinement_steps:   total refinement iterations applied to the block
                            (integer, >= 0)

    The NamedTuple representation keeps the API extensible: subclasses or
    alternative tuple sizes can be dropped in by updating ``ModelConfig.tuple_size``.
    """

    block_size: int
    stabilizing_steps: int
    refinement_steps: int


@dataclass(slots=True)
class ModelConfig:
    """Hyper-parameters that define the Transformer architecture.

    Keeping all architecture choices in one place makes it easy to run
    ablations or extend the model for different tuple types.
    """

    # number of previous tuples used as context when predicting the next one
    window_size: int = 8
    # dimensionality of the Transformer embedding space
    d_model: int = 64
    # number of parallel attention heads
    n_heads: int = 4
    # number of stacked Transformer encoder layers
    n_layers: int = 2
    # dropout probability applied in embedding and attention sublayers
    dropout: float = 0.1
    # number of integer fields in each input tuple; update when the tuple
    # structure changes (e.g. adding a fourth field)
    tuple_size: int = 3

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            msg = (
                f"ModelConfig.d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
            raise ValueError(msg)
        if self.window_size < 1:
            msg = "ModelConfig.window_size must be >= 1"
            raise ValueError(msg)
        if self.tuple_size < 1:
            msg = "ModelConfig.tuple_size must be >= 1"
            raise ValueError(msg)


@dataclass(slots=True)
class TrainConfig:
    """Hyper-parameters that control training."""

    # total gradient-update steps
    max_epochs: int = 100
    # mini-batch size
    batch_size: int = 32
    # AdamW learning rate
    learning_rate: float = 1e-3
    # AdamW weight decay
    weight_decay: float = 1e-4
    # seed for dataset split and weight initialisation
    seed: int = 42
    # fraction of the sequence used for validation (0 < val_fraction < 1)
    val_fraction: float = 0.2
    # stop early after this many epochs without validation improvement
    patience: int = 10
    # print training metrics every N epochs (0 to suppress)
    log_interval: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            msg = "TrainConfig.val_fraction must be in (0, 1)"
            raise ValueError(msg)
        if self.max_epochs < 1:
            msg = "TrainConfig.max_epochs must be >= 1"
            raise ValueError(msg)


@dataclass(slots=True)
class PredictionResult:
    """Output of a single next-tuple prediction."""

    # rounded integer prediction
    predicted_tuple: PhaseTuple
    # raw float outputs from the regression heads (before rounding)
    raw_output: list[float]
    # optional diagnostics or provenance information
    metadata: dict[str, Any] = field(default_factory=dict)
