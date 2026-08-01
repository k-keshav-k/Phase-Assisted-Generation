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
        self._safe_streak = self._safe_streak + 1 if eligible else 0
        should_stop = self._safe_streak >= self.patience
        decision = StopDecision(
            should_stop=should_stop,
            risk_score=score,
            safe_streak=self._safe_streak,
            reason="risk_certified_candidate" if should_stop else "continue",
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
