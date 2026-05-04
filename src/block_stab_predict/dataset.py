"""Dataset loading and feature-matrix construction for the block/stabilising-step predictor.

Loads ``stab_tuples_*.jsonl`` files and transforms them into flat
feature matrices and target arrays suitable for scikit-learn.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from block_stab_predict.features import compute_features, feature_names
from block_stab_predict.schema import RFConfig


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a ``stab_tuples_*.jsonl`` file.

    Each line is a JSON object with keys ``sample_id``, ``dataset``,
    ``tuples`` (a list of per-block dicts containing at least the fields
    in :data:`~block_stab_predict.schema.TUPLE_FIELDS`).

    Returns:
        List of per-sample dicts.  Empty list if the file is empty or
        contains no valid records.
    """
    samples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            tuples = record.get("tuples")
            if not isinstance(tuples, list) or len(tuples) == 0:
                continue
            samples.append(record)
    return samples


# ── Feature-matrix construction ───────────────────────────────────────


def build_X_y(
    samples: list[dict],
    config: RFConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix *X* and target matrix *Y* from a list of samples.

    For each sample, every valid position ``i >= config.window_size`` in
    the tuple sequence produces one training example: features from the
    ``config.window_size`` preceding tuples, targets from the current
    tuple.

    Args:
        samples: List of per-sample dicts as returned by :func:`load_jsonl`.
        config:  Predictor configuration.

    Returns:
        ``(X, Y, names)`` where:
            X:      float32 array of shape ``(n_examples, n_features)``.
            Y:      float32 array of shape ``(n_examples, n_targets)``
                    with columns ordered by ``config.target_fields``.
            names:  List of feature names (for interpretability).
    """
    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []

    # Validate that all tuples carry the required fields.
    required_fields = set(config.feature_fields) | set(config.target_fields)
    for sidx, sample in enumerate(samples):
        for tidx, t in enumerate(sample["tuples"]):
            missing = required_fields - set(t.keys())
            if missing:
                msg = (
                    f"Sample {sidx} ({sample.get('sample_id', '?')}), "
                    f"tuple {tidx} is missing fields: {sorted(missing)}"
                )
                raise ValueError(msg)

    for sample in samples:
        tuples = sample["tuples"]
        seq_len = len(tuples)

        if seq_len <= config.window_size:
            continue  # not enough context for even one prediction

        for i in range(config.window_size, seq_len):
            past = tuples[i - config.window_size : i]
            current = tuples[i]

            feat = compute_features(past, config)
            tgt = np.array(
                [current[f] for f in config.target_fields],
                dtype=np.float32,
            )
            X_rows.append(feat)
            Y_rows.append(tgt)

    if not X_rows:
        raise ValueError(
            f"No examples could be generated "
            f"(samples={len(samples)}, window_size={config.window_size}). "
            "Ensure every sample has at least window_size+1 tuples."
        )

    return (
        np.stack(X_rows, axis=0),
        np.stack(Y_rows, axis=0),
        feature_names(config),
    )


# ── Train / test split by sample (prevents temporal leakage) ──────────


def train_test_split_by_sample(
    samples: list[dict],
    config: RFConfig,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Split *samples* by sample ID, then build X / y for each split.

    This prevents the same sample's temporally adjacent windows from
    leaking across the train / test boundary.

    Args:
        samples:  Per-sample dicts from :func:`load_jsonl`.
        config:   Predictor configuration.
        shuffle:  Whether to shuffle sample order before splitting.
        seed:     Random seed for reproducibility (overrides
                  ``config.random_state`` when provided).

    Returns:
        ``(X_train, X_test, Y_train, Y_test, names)``.
    """
    rng = random.Random(seed if seed is not None else config.random_state)

    indices = list(range(len(samples)))
    if shuffle:
        rng.shuffle(indices)

    n_samples = len(samples)
    if n_samples < 2:
        msg = (
            f"Need at least 2 samples to split, got {n_samples}. "
            "Use build_X_y() directly to train on all data without a split."
        )
        raise ValueError(msg)

    split_idx = max(1, int(n_samples * (1.0 - config.test_size)))
    train_idxs = indices[:split_idx]
    test_idxs = indices[split_idx:]

    if not train_idxs or not test_idxs:
        msg = (
            f"Split produced an empty set "
            f"(samples={n_samples}, test_size={config.test_size}). "
            "Reduce test_size or provide more samples."
        )
        raise ValueError(msg)

    train_samples = [samples[i] for i in train_idxs]
    test_samples = [samples[i] for i in test_idxs]

    # Build feature matrices on each split independently so normalisation
    # statistics (if any) would be computed correctly per split.
    X_train, Y_train, names = build_X_y(train_samples, config)
    X_test, Y_test, _ = build_X_y(test_samples, config)

    return X_train, X_test, Y_train, Y_test, names
