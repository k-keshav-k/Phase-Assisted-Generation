from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pag.experiments.rc_pag_features import (
    RealizedBlock,
    StepObservation,
    extract_features,
    feature_names,
    vectorize_features,
)


class RiskScorer(Protocol):
    def predict_risk(self, features: Mapping[str, float]) -> float: ...


class BenefitScorer(Protocol):
    def predict_remaining_nfe(self, features: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, slots=True)
class TrainingExample:
    features: Mapping[str, float]
    unsafe: bool
    prompt_id: str

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("training example prompt_id must be non-empty")


@dataclass(frozen=True, slots=True)
class BenefitExample:
    features: Mapping[str, float]
    remaining_nfe: float
    prompt_id: str

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("benefit example prompt_id must be non-empty")
        if not math.isfinite(self.remaining_nfe) or self.remaining_nfe < 0.0:
            raise ValueError("remaining NFE target must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NormalizedNFEReductionExample:
    features: Mapping[str, float]
    nfe_reduction: float
    prompt_id: str

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("normalized NFE example prompt_id must be non-empty")
        if not math.isfinite(self.nfe_reduction) or not 0.0 <= self.nfe_reduction <= 1.0:
            raise ValueError("normalized NFE reduction target must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class StopDecision:
    should_stop: bool
    risk_score: float
    safe_streak: int
    reason: str
    predicted_nfe_savings: float = 0.0
    temporal_js: float = 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RiskEstimator:
    def __init__(
        self,
        *,
        kind: str,
        include_history: bool,
        history_window: int,
        names: Sequence[str],
        model: Any | None,
        constant_risk: float | None,
    ) -> None:
        self.kind = str(kind)
        self.include_history = bool(include_history)
        self.history_window = int(history_window)
        self.names = tuple(str(name) for name in names)
        self.model = model
        self.constant_risk = constant_risk

    @classmethod
    def fit(
        cls,
        examples: Sequence[TrainingExample],
        *,
        kind: str,
        include_history: bool,
        history_window: int,
        seed: int,
    ) -> RiskEstimator:
        if not examples:
            raise ValueError("risk estimator requires training examples")
        if kind not in {"hist_gradient_boosting", "logistic"}:
            raise ValueError(f"unsupported risk estimator: {kind}")
        if history_window < 1:
            raise ValueError("history_window must be positive")
        names = feature_names(include_history=include_history)
        features = np.stack([vectorize_features(example.features, names) for example in examples])
        labels = np.asarray([int(example.unsafe) for example in examples], dtype=np.int64)
        unique = np.unique(labels)
        if unique.size == 1:
            return cls(
                kind=kind,
                include_history=include_history,
                history_window=history_window,
                names=names,
                model=None,
                constant_risk=float(unique[0]),
            )
        if kind == "logistic":
            model: Any = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=seed),
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=8,
                learning_rate=0.05,
                l2_regularization=0.1,
                random_state=seed,
            )
        model.fit(features, labels)
        return cls(
            kind=kind,
            include_history=include_history,
            history_window=history_window,
            names=names,
            model=model,
            constant_risk=None,
        )

    def predict_risk(self, features: Mapping[str, float]) -> float:
        if self.constant_risk is not None:
            return self.constant_risk
        if self.model is None:
            raise RuntimeError("risk estimator has neither a model nor a constant score")
        row = vectorize_features(features, self.names)[None, :]
        probabilities = np.asarray(self.model.predict_proba(row)[0], dtype=np.float64)
        classes = list(self.model.classes_)
        risk = float(probabilities[classes.index(1)])
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError("risk estimator produced an invalid probability")
        return risk

    def save(self, path: str | Path) -> dict[str, object]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": self.kind,
                "include_history": self.include_history,
                "history_window": self.history_window,
                "names": self.names,
                "model": self.model,
                "constant_risk": self.constant_risk,
            },
            destination,
        )
        sha256 = _sha256_file(destination)
        metadata: dict[str, object] = {
            "schema_version": 1,
            "sha256": sha256,
            "kind": self.kind,
            "include_history": self.include_history,
            "history_window": self.history_window,
            "feature_names": list(self.names),
        }
        metadata_path = destination.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> RiskEstimator:
        source = Path(path)
        metadata_path = source.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sha256") != _sha256_file(source):
                raise ValueError("risk estimator hash does not match metadata")
        payload: dict[str, Any] = joblib.load(source)
        return cls(
            kind=str(payload["kind"]),
            include_history=bool(payload["include_history"]),
            history_window=int(payload["history_window"]),
            names=tuple(str(name) for name in payload["names"]),
            model=payload["model"],
            constant_risk=(
                None if payload["constant_risk"] is None else float(payload["constant_risk"])
            ),
        )


class CalibratedRiskEstimator:
    """Local disagreement scorer with a prompt-disjoint isotonic calibration map."""

    def __init__(
        self,
        *,
        base: RiskEstimator,
        calibrator: Any | None,
        constant_risk: float | None,
        training_prompt_ids: Sequence[str],
        calibration_prompt_ids: Sequence[str],
    ) -> None:
        self.base = base
        self.calibrator = calibrator
        self.constant_risk = constant_risk
        self.training_prompt_ids = tuple(str(value) for value in training_prompt_ids)
        self.calibration_prompt_ids = tuple(str(value) for value in calibration_prompt_ids)
        self.kind = base.kind
        self.include_history = base.include_history
        self.history_window = base.history_window
        self.names = base.names

    @classmethod
    def fit(
        cls,
        *,
        training_examples: Sequence[TrainingExample],
        calibration_examples: Sequence[TrainingExample],
        kind: str,
        include_history: bool,
        history_window: int,
        seed: int,
    ) -> CalibratedRiskEstimator:
        if not training_examples or not calibration_examples:
            raise ValueError("calibrated risk estimator requires training and calibration examples")
        training_prompt_ids = tuple(sorted({example.prompt_id for example in training_examples}))
        calibration_prompt_ids = tuple(
            sorted({example.prompt_id for example in calibration_examples})
        )
        if set(training_prompt_ids) & set(calibration_prompt_ids):
            raise ValueError("risk training and calibration must remain prompt-disjoint")
        base = RiskEstimator.fit(
            training_examples,
            kind=kind,
            include_history=include_history,
            history_window=history_window,
            seed=seed,
        )
        raw_scores = np.asarray(
            [base.predict_risk(example.features) for example in calibration_examples],
            dtype=np.float64,
        )
        labels = np.asarray(
            [int(example.unsafe) for example in calibration_examples],
            dtype=np.float64,
        )
        if np.unique(labels).size == 1:
            calibrator = None
            constant_risk = float(labels[0])
        else:
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(raw_scores, labels)
            constant_risk = None
        return cls(
            base=base,
            calibrator=calibrator,
            constant_risk=constant_risk,
            training_prompt_ids=training_prompt_ids,
            calibration_prompt_ids=calibration_prompt_ids,
        )

    def predict_risk(self, features: Mapping[str, float]) -> float:
        if self.constant_risk is not None:
            return self.constant_risk
        if self.calibrator is None:
            raise RuntimeError("calibrated risk estimator has no calibration map")
        raw = self.base.predict_risk(features)
        risk = float(self.calibrator.predict(np.asarray([raw], dtype=np.float64))[0])
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError("calibrated risk estimator produced an invalid probability")
        return risk

    def save(self, path: str | Path) -> dict[str, object]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "base": self.base,
                "calibrator": self.calibrator,
                "constant_risk": self.constant_risk,
                "training_prompt_ids": self.training_prompt_ids,
                "calibration_prompt_ids": self.calibration_prompt_ids,
            },
            destination,
        )
        metadata: dict[str, object] = {
            "schema_version": 1,
            "sha256": _sha256_file(destination),
            "kind": self.kind,
            "target": "local_full_trajectory_disagreement",
            "calibration": "isotonic_prompt_holdout",
            "constant_calibration": self.constant_risk,
            "include_history": self.include_history,
            "history_window": self.history_window,
            "feature_names": list(self.names),
            "training_prompts": len(self.training_prompt_ids),
            "calibration_prompts": len(self.calibration_prompt_ids),
        }
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> CalibratedRiskEstimator:
        source = Path(path)
        metadata_path = source.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sha256") != _sha256_file(source):
                raise ValueError("calibrated risk estimator hash does not match metadata")
            if metadata.get("target") != "local_full_trajectory_disagreement":
                raise ValueError("calibrated risk estimator metadata has the wrong target")
        payload: dict[str, Any] = joblib.load(source)
        return cls(
            base=payload["base"],
            calibrator=payload["calibrator"],
            constant_risk=(
                None if payload["constant_risk"] is None else float(payload["constant_risk"])
            ),
            training_prompt_ids=payload["training_prompt_ids"],
            calibration_prompt_ids=payload["calibration_prompt_ids"],
        )


class RemainingNFEEstimator:
    def __init__(
        self,
        *,
        include_history: bool,
        history_window: int,
        names: Sequence[str],
        model: Any,
    ) -> None:
        self.include_history = bool(include_history)
        self.history_window = int(history_window)
        self.names = tuple(str(name) for name in names)
        self.model = model

    @classmethod
    def fit(
        cls,
        examples: Sequence[BenefitExample],
        *,
        include_history: bool,
        history_window: int,
        seed: int,
    ) -> RemainingNFEEstimator:
        if not examples:
            raise ValueError("remaining-NFE estimator requires training examples")
        names = feature_names(include_history=include_history)
        features = np.stack([vectorize_features(example.features, names) for example in examples])
        targets = np.asarray([example.remaining_nfe for example in examples], dtype=np.float64)
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=8,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed,
            loss="squared_error",
        )
        model.fit(features, targets)
        return cls(
            include_history=include_history,
            history_window=history_window,
            names=names,
            model=model,
        )

    def predict_remaining_nfe(self, features: Mapping[str, float]) -> float:
        value = float(self.model.predict(vectorize_features(features, self.names)[None, :])[0])
        if not math.isfinite(value):
            raise ValueError("remaining-NFE estimator produced a non-finite value")
        return max(0.0, value)

    def save(self, path: str | Path) -> dict[str, object]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "include_history": self.include_history,
                "history_window": self.history_window,
                "names": self.names,
                "model": self.model,
            },
            destination,
        )
        metadata: dict[str, object] = {
            "schema_version": 1,
            "sha256": _sha256_file(destination),
            "kind": "hist_gradient_boosting_regressor",
            "target": "remaining_nfe",
            "include_history": self.include_history,
            "history_window": self.history_window,
            "feature_names": list(self.names),
        }
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> RemainingNFEEstimator:
        source = Path(path)
        metadata_path = source.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sha256") != _sha256_file(source):
                raise ValueError("remaining-NFE estimator hash does not match metadata")
        payload: dict[str, Any] = joblib.load(source)
        return cls(
            include_history=bool(payload["include_history"]),
            history_window=int(payload["history_window"]),
            names=tuple(str(name) for name in payload["names"]),
            model=payload["model"],
        )


class NormalizedNFEReductionEstimator:
    """Predict prompt-normalized NFE reduction from compact online features."""

    def __init__(
        self,
        *,
        include_history: bool,
        history_window: int,
        names: Sequence[str],
        model: Any,
    ) -> None:
        self.include_history = bool(include_history)
        self.history_window = int(history_window)
        self.names = tuple(str(name) for name in names)
        self.model = model

    @classmethod
    def fit(
        cls,
        examples: Sequence[NormalizedNFEReductionExample],
        *,
        include_history: bool,
        history_window: int,
        seed: int,
    ) -> NormalizedNFEReductionEstimator:
        if not examples:
            raise ValueError("normalized NFE estimator requires training examples")
        names = feature_names(include_history=include_history)
        features = np.stack([vectorize_features(example.features, names) for example in examples])
        targets = np.asarray([example.nfe_reduction for example in examples], dtype=np.float64)
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=8,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed,
            loss="squared_error",
        )
        model.fit(features, targets)
        return cls(
            include_history=include_history,
            history_window=history_window,
            names=names,
            model=model,
        )

    def predict_remaining_nfe(self, features: Mapping[str, float]) -> float:
        value = float(self.model.predict(vectorize_features(features, self.names)[None, :])[0])
        if not math.isfinite(value):
            raise ValueError("normalized NFE estimator produced a non-finite value")
        return min(1.0, max(0.0, value))

    def save(self, path: str | Path) -> dict[str, object]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "include_history": self.include_history,
                "history_window": self.history_window,
                "names": self.names,
                "model": self.model,
            },
            destination,
        )
        metadata: dict[str, object] = {
            "schema_version": 1,
            "sha256": _sha256_file(destination),
            "kind": "hist_gradient_boosting_regressor",
            "target": "normalized_nfe_reduction",
            "bounds": [0.0, 1.0],
            "include_history": self.include_history,
            "history_window": self.history_window,
            "feature_names": list(self.names),
        }
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> NormalizedNFEReductionEstimator:
        source = Path(path)
        metadata_path = source.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sha256") != _sha256_file(source):
                raise ValueError("normalized NFE estimator hash does not match metadata")
            if metadata.get("target") != "normalized_nfe_reduction":
                raise ValueError("normalized NFE estimator metadata has the wrong target")
        payload: dict[str, Any] = joblib.load(source)
        return cls(
            include_history=bool(payload["include_history"]),
            history_window=int(payload["history_window"]),
            names=tuple(str(name) for name in payload["names"]),
            model=payload["model"],
        )


class RiskStoppingPolicy:
    def __init__(
        self,
        scorer: RiskScorer,
        *,
        threshold: float,
        min_steps: int,
        patience: int,
        include_history: bool = True,
        history_window: int = 8,
        max_remaining_fraction: float = 1.0,
        benefit_scorer: BenefitScorer | None = None,
        min_predicted_nfe_savings: float = 0.0,
        max_temporal_js: float = 1.0,
        require_exact_agreement: bool = False,
        force_full_budget: bool = False,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if min_steps < 1 or patience < 1 or history_window < 1:
            raise ValueError("min_steps, patience, and history_window must be positive")
        if not 0.0 < max_remaining_fraction <= 1.0:
            raise ValueError("max_remaining_fraction must be in (0, 1]")
        if not math.isfinite(min_predicted_nfe_savings) or min_predicted_nfe_savings < 0.0:
            raise ValueError("min_predicted_nfe_savings must be finite and non-negative")
        if not 0.0 <= max_temporal_js <= 1.0:
            raise ValueError("max_temporal_js must be in [0, 1]")
        self.scorer = scorer
        self.threshold = float(threshold)
        self.min_steps = int(min_steps)
        self.patience = int(patience)
        self.include_history = bool(include_history)
        self.history_window = int(history_window)
        self.max_remaining_fraction = float(max_remaining_fraction)
        self.benefit_scorer = benefit_scorer
        self.min_predicted_nfe_savings = float(min_predicted_nfe_savings)
        self.max_temporal_js = float(max_temporal_js)
        self.require_exact_agreement = bool(require_exact_agreement)
        self.force_full_budget = bool(force_full_budget)
        self.reset_prompt()

    @classmethod
    def full_budget(cls) -> RiskStoppingPolicy:
        return cls(
            _ConstantScorer(1.0),
            threshold=0.0,
            min_steps=1,
            patience=1,
            include_history=False,
            max_remaining_fraction=1.0,
            force_full_budget=True,
        )

    @property
    def history(self) -> tuple[RealizedBlock, ...]:
        return tuple(self._history)

    @property
    def decision_trace(self) -> tuple[StopDecision, ...]:
        return tuple(self._decision_trace)

    def reset_prompt(self) -> None:
        self._history: list[RealizedBlock] = []
        self._decision_trace: list[StopDecision] = []
        self.start_block()

    def start_block(self) -> None:
        self._previous: StepObservation | None = None
        self._safe_streak = 0
        self._pending_tokens: tuple[int, ...] | None = None

    def record_realized(self, block: RealizedBlock) -> None:
        self._history.append(block)

    def observe(self, observation: StepObservation) -> StopDecision:
        if self.force_full_budget:
            decision = StopDecision(False, 1.0, 0, "full_budget_fallback")
            self._previous = observation
            self._decision_trace.append(decision)
            return decision
        all_features = extract_features(
            observation,
            previous=self._previous,
            history=self._history if self.include_history else (),
            history_window=self.history_window,
        )
        names = feature_names(include_history=self.include_history)
        selected_features = {name: all_features[name] for name in names}
        score = float(self.scorer.predict_risk(selected_features))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("risk scorer must return a probability in [0, 1]")
        remaining_fraction = sum(observation.masked) / observation.block_size
        predicted_savings = (
            float(self.benefit_scorer.predict_remaining_nfe(selected_features))
            if self.benefit_scorer is not None
            else 0.0
        )
        if not math.isfinite(predicted_savings) or predicted_savings < 0.0:
            raise ValueError("benefit scorer must return a finite non-negative value")
        masked_js = [
            value
            for value, masked in zip(observation.temporal_js, observation.masked, strict=True)
            if masked
        ]
        temporal_js = max(masked_js, default=0.0)
        eligible = (
            observation.step_index >= self.min_steps
            and remaining_fraction <= self.max_remaining_fraction
            and score <= self.threshold
            and temporal_js <= self.max_temporal_js
            and (self.benefit_scorer is None or predicted_savings >= self.min_predicted_nfe_savings)
        )
        reason = "continue"
        if self.require_exact_agreement:
            if not eligible:
                self._safe_streak = 0
                self._pending_tokens = None
                should_stop = False
            else:
                proposed_tokens = tuple(observation.token_ids)
                agreement = self._pending_tokens is not None and all(
                    not is_masked or previous == current
                    for is_masked, previous, current in zip(
                        observation.masked,
                        self._pending_tokens or (),
                        proposed_tokens,
                        strict=True,
                    )
                )
                if agreement:
                    self._safe_streak += 1
                    should_stop = self._safe_streak >= self.patience
                    reason = "agreement_verified" if should_stop else "pending_verification"
                else:
                    reason = (
                        "proposal_changed"
                        if self._pending_tokens is not None
                        else "pending_verification"
                    )
                    self._safe_streak = 1
                    should_stop = False
                self._pending_tokens = proposed_tokens
        else:
            self._safe_streak = self._safe_streak + 1 if eligible else 0
            should_stop = self._safe_streak >= self.patience
            reason = "risk_certified_candidate" if should_stop else "continue"
        decision = StopDecision(
            should_stop=should_stop,
            risk_score=score,
            safe_streak=self._safe_streak,
            reason=reason,
            predicted_nfe_savings=predicted_savings,
            temporal_js=temporal_js,
        )
        self._previous = observation
        self._decision_trace.append(decision)
        return decision


class _ConstantScorer:
    def __init__(self, risk: float) -> None:
        self.risk = float(risk)

    def predict_risk(self, features: Mapping[str, float]) -> float:
        del features
        return self.risk
