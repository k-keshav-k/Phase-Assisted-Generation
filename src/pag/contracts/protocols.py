from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pag.contracts.artifacts import (
    AdaptiveRunArtifacts,
    BaselineRunArtifacts,
    EvaluationArtifacts,
    PhaseArtifacts,
)
from pag.contracts.schemas import RunConfig, SampleRecord


class BaselineRunner(Protocol):
    def __call__(
        self,
        run_config: RunConfig,
        samples: Sequence[SampleRecord],
    ) -> BaselineRunArtifacts:
        ...


class PhaseRunner(Protocol):
    def __call__(
        self,
        run_config: RunConfig,
        baseline_artifacts: BaselineRunArtifacts,
        hidden_state_features: dict[str, list[dict[str, float]]] | None = None,
    ) -> PhaseArtifacts:
        ...


class SchedulerRunner(Protocol):
    def __call__(
        self,
        run_config: RunConfig,
        baseline_artifacts: BaselineRunArtifacts,
        phase_artifacts: PhaseArtifacts,
    ) -> AdaptiveRunArtifacts:
        ...


class Evaluator(Protocol):
    def __call__(
        self,
        run_config: RunConfig,
        baseline_artifacts: BaselineRunArtifacts,
        adaptive_artifacts: AdaptiveRunArtifacts,
    ) -> EvaluationArtifacts:
        ...


class ArtifactStore(Protocol):
    def save(self, target: Path, payload: Any) -> None:
        ...
