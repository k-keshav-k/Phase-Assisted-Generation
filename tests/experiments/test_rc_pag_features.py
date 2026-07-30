from __future__ import annotations

import math

import numpy as np
import pytest

from pag.experiments.rc_pag_features import (
    RealizedBlock,
    StepObservation,
    extract_features,
    feature_names,
    vectorize_features,
)


def observation(*, step: int, token_ids: list[int]) -> StepObservation:
    return StepObservation.from_arrays(
        step_index=step,
        block_size=4,
        masked=[True, True, False, False],
        top1_probs=[0.6, 0.7, 0.9, 0.8],
        top2_probs=[0.2, 0.3, 0.1, 0.2],
        entropies=[1.0, 0.8, 0.2, 0.4],
        token_ids=token_ids,
        digit_ids={9},
        delimiter_ids={4},
    )


def test_step_observation_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="top1 probabilities"):
        StepObservation.from_arrays(
            step_index=1,
            block_size=4,
            masked=[True, True, False, False],
            top1_probs=[1.2, 0.5, 0.9, 0.8],
            top2_probs=[0.1, 0.2, 0.1, 0.1],
            entropies=[0.1, 0.2, 0.3, 0.4],
            token_ids=[1, 2, 3, 4],
        )


def test_step_observation_rejects_shape_and_probability_order() -> None:
    with pytest.raises(ValueError, match="match block_size"):
        StepObservation.from_arrays(
            step_index=1,
            block_size=4,
            masked=[True],
            top1_probs=[0.7],
            top2_probs=[0.2],
            entropies=[0.5],
            token_ids=[1],
        )
    with pytest.raises(ValueError, match="top2 probabilities cannot exceed top1"):
        StepObservation.from_arrays(
            step_index=1,
            block_size=1,
            masked=[True],
            top1_probs=[0.2],
            top2_probs=[0.3],
            entropies=[0.5],
            token_ids=[1],
        )


def test_feature_vector_contains_local_and_history_fields() -> None:
    previous = observation(step=1, token_ids=[1, 2, 3, 4])
    current = observation(step=2, token_ids=[1, 9, 3, 4])
    history = [RealizedBlock(4, 3, 0.9, 0.6, 0.25, 0.0)]
    features = extract_features(current, previous=previous, history=history, history_window=4)

    assert features["local.token_churn"] == 0.25
    assert features["local.remaining_fraction"] == 0.5
    assert features["local.entropy_mean"] == 0.9
    assert features["local.margin_min"] == pytest.approx(0.4)
    assert features["local.digit_fraction"] == 0.25
    assert features["local.delimiter_fraction"] == 0.25
    assert features["history.length"] == 1.0
    assert features["history.nfe_last"] == 3.0
    assert all(math.isfinite(value) for value in features.values())

    names = feature_names(include_history=True)
    row = vectorize_features(features, names)
    assert row.dtype == np.float64
    assert row.shape == (len(names),)
    assert np.isfinite(row).all()


def test_local_feature_names_exclude_history() -> None:
    names = feature_names(include_history=False)
    assert names
    assert all(name.startswith("local.") for name in names)


def test_empty_mask_and_history_produce_finite_zero_summaries() -> None:
    current = StepObservation.from_arrays(
        step_index=3,
        block_size=2,
        masked=[False, False],
        top1_probs=[0.9, 0.8],
        top2_probs=[0.05, 0.1],
        entropies=[0.2, 0.3],
        token_ids=[1, 2],
    )
    features = extract_features(current, previous=None, history=(), history_window=4)
    assert features["local.remaining_count"] == 0.0
    assert features["local.entropy_mean"] == 0.0
    assert features["history.length"] == 0.0
    assert all(math.isfinite(value) for value in features.values())
