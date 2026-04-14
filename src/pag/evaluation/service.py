from __future__ import annotations

from pag.contracts.artifacts import AdaptiveRunArtifacts, BaselineRunArtifacts, EvaluationArtifacts
from pag.contracts.protocols import Evaluator
from pag.contracts.schemas import RunConfig
from pag.evaluation.stubs import mock_evaluator
from pag.orchestration.registry import get_evaluator


def evaluate_runs(
    run_config: RunConfig,
    baseline_artifacts: BaselineRunArtifacts,
    adaptive_artifacts: AdaptiveRunArtifacts,
    implementation: Evaluator | None = None,
) -> EvaluationArtifacts:
    evaluator = implementation or get_evaluator(run_config.evaluation.name) or mock_evaluator
    return evaluator(run_config, baseline_artifacts, adaptive_artifacts)

