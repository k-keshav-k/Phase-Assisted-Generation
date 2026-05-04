"""Config and field constants for the block/stabilising-step predictor.

The model predicts (block_size, max_stab_step) for the next generation block
given rolling-window statistics of past realised blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Field-name constants ──────────────────────────────────────────────
# These mirror the keys in each tuple dict inside the stab_tuples_*.jsonl files.

TUPLE_FIELDS: tuple[str, ...] = (
    "block_size",
    "nfe",
    "mean_stab_step",
    "max_stab_step",
    "mean_ref_step",
    "max_ref_step",
    "mean_gap",
    "max_gap",
)

# Past-tuple fields used as regression features.
# Phase 2: all fields available in trace data.
# During online inference the generation loop computes these via per-block
# tracking and passes them to ``InferencePredictor.record()`` as extra fields.
FEATURE_FIELDS: tuple[str, ...] = (
    "nfe",
    "block_size",
    "max_stab_step",
    "mean_ref_step",
    "mean_gap",
    "max_gap",
)

# Target fields for multi-output regression.
TARGET_FIELDS: tuple[str, ...] = ("block_size", "max_stab_step")

# Statistics computed over a sliding window for each feature field.
# This is intentionally a module-level constant rather than a config field:
# the set of stats is inherent to the feature-engineering strategy, not a
# hyper-parameter we expect to ablate.
FIELD_STATS: tuple[str, ...] = ("last", "mean", "std", "min", "max", "trend")

# ── Configuration ─────────────────────────────────────────────────────


@dataclass(slots=True)
class RFConfig:
    """Hyper-parameters for the Random Forest predictor.

    Attributes
    ----------
    window_size:
        Number of past blocks used as context when predicting the next one.
    feature_fields:
        Subset of ``TUPLE_FIELDS`` used to build rolling-window features.
    target_fields:
        Subset of ``TUPLE_FIELDS`` to predict.
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum tree depth.  ``None`` means unlimited.
    min_samples_leaf:
        Minimum samples required at a leaf node.
    random_state:
        Seed for reproducibility.
    n_jobs:
        Number of parallel jobs (``-1`` = all CPUs).
    test_size:
        Fraction of **samples** (not tuples) held out for validation.
    """

    # Feature engineering
    window_size: int = 20
    feature_fields: tuple[str, ...] = field(default_factory=lambda: FEATURE_FIELDS)
    target_fields: tuple[str, ...] = field(default_factory=lambda: TARGET_FIELDS)

    # Forest structure
    n_estimators: int = 200
    max_depth: int | None = 15
    min_samples_leaf: int = 5

    # Reproducibility & resources
    random_state: int = 42
    n_jobs: int = -1

    # Validation split
    test_size: float = 0.2

    def __post_init__(self) -> None:
        if self.window_size < 1:
            msg = f"window_size must be >= 1, got {self.window_size}"
            raise ValueError(msg)
        if not self.feature_fields:
            msg = "feature_fields must not be empty"
            raise ValueError(msg)
        if not self.target_fields:
            msg = "target_fields must not be empty"
            raise ValueError(msg)
        if not 0.0 < self.test_size < 1.0:
            msg = f"test_size must be in (0, 1), got {self.test_size}"
            raise ValueError(msg)
