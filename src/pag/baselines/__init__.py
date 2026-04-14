from __future__ import annotations

from collections.abc import Sequence

from pag.baselines.stubs import mock_baseline_runner
from pag.contracts.artifacts import BaselineRunArtifacts
from pag.contracts.protocols import BaselineRunner
from pag.contracts.schemas import RunConfig, SampleRecord

__all__ = ["run_baseline"]


def run_baseline(
    run_config: RunConfig,
    samples: Sequence[SampleRecord],
    implementation: BaselineRunner | None = None,
) -> BaselineRunArtifacts:
    runner = implementation or mock_baseline_runner
    return runner(run_config, samples)
