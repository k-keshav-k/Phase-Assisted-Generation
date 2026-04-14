from __future__ import annotations

from collections.abc import Sequence

from pag.baselines.stubs import mock_baseline_runner
from pag.contracts.artifacts import BaselineRunArtifacts
from pag.contracts.protocols import BaselineRunner
from pag.contracts.schemas import RunConfig, SampleRecord
from pag.orchestration.registry import get_baseline_runner


def run_baseline(
    run_config: RunConfig,
    samples: Sequence[SampleRecord],
    implementation: BaselineRunner | None = None,
) -> BaselineRunArtifacts:
    runner = implementation or get_baseline_runner(run_config.model.name) or mock_baseline_runner
    return runner(run_config, samples)

