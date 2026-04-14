from __future__ import annotations

from pag.contracts.artifacts import AdaptiveRunArtifacts, BaselineRunArtifacts, PhaseArtifacts
from pag.contracts.protocols import SchedulerRunner
from pag.contracts.schemas import RunConfig
from pag.scheduler.stubs import mock_scheduler_runner

__all__ = ["run_adaptive_decoding"]


def run_adaptive_decoding(
    run_config: RunConfig,
    baseline_artifacts: BaselineRunArtifacts,
    phase_artifacts: PhaseArtifacts,
    implementation: SchedulerRunner | None = None,
) -> AdaptiveRunArtifacts:
    runner = implementation or mock_scheduler_runner
    return runner(run_config, baseline_artifacts, phase_artifacts)
