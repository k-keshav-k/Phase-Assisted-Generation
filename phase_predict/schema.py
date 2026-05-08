"""Data schemas for the phase_predict package.

All public data structures are defined here to keep the rest of the package
decoupled from each other and easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


class PhaseTuple(NamedTuple):
    """One generation-block's adaptive parameters (OUTPUT format).

    Fields map directly to the scheduling decisions produced by the PAG
    scheduler:
      - block_size:        number of tokens decoded together in this block
      - refinement_steps:  total refinement iterations applied to the block
                           (integer, >= 0)

    The NamedTuple representation keeps the API extensible: additional fields
    (e.g. stabilizing_steps, temperature) can be appended here and picked up
    by downstream consumers.
    """

    block_size: int
    refinement_steps: int


@dataclass(slots=True)
class ExtendedPhaseTuple:
    """Extended phase tuple with multiple input features for training.

    This structure holds all available input features that can be used to
    predict the output (block_size, refinement_steps). The model trains on
    all these input fields but outputs only the standard 2-field PhaseTuple.

    To add new features in the future, simply add them as new fields here
    and pass them during data loading via ``feature_fields`` parameter.

    Example fields:
      - block_size: number of tokens in the block
      - nfe: number of function evaluations
      - max_stab_step: maximum stabilization step
      - stabilizing_entropy: entropy during stabilization
      - (add more as needed)
    """

    values: dict[str, float]

    def __getitem__(self, field_name: str) -> float:
        return self.values.get(field_name, 0.0)

    def __len__(self) -> int:
        return len(self.values)

    def as_list(self, field_names: list[str]) -> list[float]:
        return [self.values.get(name, 0.0) for name in field_names]


@dataclass(slots=True)
class ModelConfig:
    """Hyper-parameters that define the Transformer architecture.

    Keeping all architecture choices in one place makes it easy to run
    ablations or extend the model for different tuple types.

    The model supports multi-feature input training:
      - input_tuple_size:  number of input features (e.g. block_size, nfe,
                          max_stab_step, etc.)

    To add more input features, simply update ``input_tuple_size`` and pass
    the corresponding field names during data loading.
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
    # number of input features in each input tuple (e.g. block_size, nfe,
    # max_stab_step, etc.). Update when adding new input features.
    input_tuple_size: int = 2
    # [DEPRECATED] kept for backward compatibility; use input_tuple_size instead
    tuple_size: int | None = None
    # number of ordinal thresholds for max_stab_step prediction (P(>k) for k=0..num_stab_thresholds-1)
    num_stab_thresholds: int = 83

    def __post_init__(self) -> None:
        # Handle backward compatibility: if tuple_size is set, use it for input_tuple_size
        if self.tuple_size is not None and self.input_tuple_size == 2:
            self.input_tuple_size = self.tuple_size

        if self.d_model % self.n_heads != 0:
            msg = (
                f"ModelConfig.d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
            raise ValueError(msg)
        if self.window_size < 1:
            msg = "ModelConfig.window_size must be >= 1"
            raise ValueError(msg)
        if self.input_tuple_size < 1:
            msg = "ModelConfig.input_tuple_size must be >= 1"
            raise ValueError(msg)
        if self.num_stab_thresholds < 1:
            msg = "ModelConfig.num_stab_thresholds must be >= 1"
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
    # maximum gradient norm for gradient clipping (0 disables clipping)
    max_grad_norm: float = 1.0
    # print training metrics every N epochs (0 to suppress)
    log_interval: int = 10
    # number of DataLoader worker processes (0 = main process only)
    num_workers: int = 4

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
