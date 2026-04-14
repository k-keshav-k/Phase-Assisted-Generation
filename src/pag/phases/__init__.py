from __future__ import annotations

from pag.contracts.artifacts import BaselineRunArtifacts, PhaseArtifacts
from pag.contracts.protocols import PhaseRunner
from pag.contracts.schemas import RunConfig
from pag.phases.stubs import mock_phase_runner

__all__ = ["run_phase_analysis"]


def run_phase_analysis(
    run_config: RunConfig,
    baseline_artifacts: BaselineRunArtifacts,
    hidden_state_features: dict[str, list[dict[str, float]]] | None = None,
    implementation: PhaseRunner | None = None,
) -> PhaseArtifacts:
    runner = implementation or mock_phase_runner
    return runner(run_config, baseline_artifacts, hidden_state_features=hidden_state_features)
