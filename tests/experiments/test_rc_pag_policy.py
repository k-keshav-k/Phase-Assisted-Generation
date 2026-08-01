from __future__ import annotations

import json

import pytest

from pag.experiments.rc_pag_features import RealizedBlock, StepObservation, extract_features
from pag.experiments.rc_pag_policy import (
    RiskEstimator,
    RiskStoppingPolicy,
    TrainingExample,
)


def observation(
    *, step: int, top1: float = 0.9, masked: list[bool] | None = None
) -> StepObservation:
    return StepObservation.from_arrays(
        step_index=step,
        block_size=2,
        masked=masked or [True, False],
        top1_probs=[top1, 0.95],
        top2_probs=[0.05, 0.03],
        entropies=[0.4, 0.1],
        token_ids=[10, 11],
    )


class FixedScorer:
    def __init__(self, *scores: float) -> None:
        self.scores = list(scores)
        self.calls: list[dict[str, float]] = []

    def predict_risk(self, features: dict[str, float]) -> float:
        self.calls.append(features)
        return self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]


def test_policy_requires_minimum_steps_and_patience() -> None:
    policy = RiskStoppingPolicy(FixedScorer(0.02), threshold=0.05, min_steps=2, patience=2)
    assert not policy.observe(observation(step=1)).should_stop
    assert not policy.observe(observation(step=2)).should_stop
    decision = policy.observe(observation(step=3))
    assert decision.should_stop
    assert decision.reason == "risk_certified_candidate"
    assert decision.safe_streak == 2


def test_policy_resets_streak_after_risky_step() -> None:
    policy = RiskStoppingPolicy(
        FixedScorer(0.01, 0.2, 0.01, 0.01),
        threshold=0.05,
        min_steps=1,
        patience=2,
    )
    assert not policy.observe(observation(step=1)).should_stop
    assert policy.observe(observation(step=2)).safe_streak == 0
    assert not policy.observe(observation(step=3)).should_stop
    assert policy.observe(observation(step=4)).should_stop


def test_policy_only_stops_in_declared_tail() -> None:
    policy = RiskStoppingPolicy(
        FixedScorer(0.0),
        threshold=0.5,
        min_steps=1,
        patience=1,
        include_history=False,
        max_remaining_fraction=0.5,
    )

    early = policy.observe(observation(step=2, masked=[True, True]))
    tail = policy.observe(observation(step=3, masked=[True, False]))

    assert not early.should_stop
    assert tail.should_stop


def test_policy_fallback_never_stops_early() -> None:
    policy = RiskStoppingPolicy.full_budget()
    for step in range(1, 9):
        assert not policy.observe(observation(step=step)).should_stop


def test_policy_supplies_phase_history_and_resets_blocks() -> None:
    scorer = FixedScorer(0.5)
    policy = RiskStoppingPolicy(
        scorer,
        threshold=0.1,
        min_steps=1,
        patience=1,
        include_history=True,
    )
    policy.record_realized(RealizedBlock(2, 3, 0.9, 0.8, 0.0, 0.5))
    policy.start_block()
    policy.observe(observation(step=1))
    assert scorer.calls[-1]["history.length"] == 1.0
    policy.reset_prompt()
    policy.observe(observation(step=1))
    assert scorer.calls[-1]["history.length"] == 0.0


def test_estimator_round_trip_and_constant_labels(tmp_path) -> None:
    rows: list[TrainingExample] = []
    history: list[RealizedBlock] = []
    for step, unsafe in ((1, True), (2, True), (3, False), (4, False)):
        features = extract_features(
            observation(step=step, top1=0.5 + step / 10),
            previous=None,
            history=history,
            history_window=4,
        )
        rows.append(TrainingExample(features=features, unsafe=unsafe, prompt_id=f"p-{step}"))

    estimator = RiskEstimator.fit(
        rows,
        kind="logistic",
        include_history=False,
        history_window=4,
        seed=7,
    )
    before = estimator.predict_risk(rows[0].features)
    path = tmp_path / "risk.joblib"
    metadata = estimator.save(path)
    after = RiskEstimator.load(path).predict_risk(rows[0].features)
    assert after == pytest.approx(before)
    assert len(metadata["sha256"]) == 64
    assert json.loads(path.with_suffix(".json").read_text())["sha256"] == metadata["sha256"]

    constant = RiskEstimator.fit(
        [TrainingExample(rows[0].features, False, "safe")],
        kind="hist_gradient_boosting",
        include_history=True,
        history_window=4,
        seed=7,
    )
    assert constant.predict_risk(rows[0].features) == 0.0


def test_policy_rejects_invalid_parameters_and_scorer_output() -> None:
    with pytest.raises(ValueError, match="threshold"):
        RiskStoppingPolicy(FixedScorer(0.1), threshold=1.1, min_steps=1, patience=1)
    policy = RiskStoppingPolicy(FixedScorer(float("nan")), threshold=0.5, min_steps=1, patience=1)
    with pytest.raises(ValueError, match="probability"):
        policy.observe(observation(step=1))
