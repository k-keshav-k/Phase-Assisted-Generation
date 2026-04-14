from __future__ import annotations

from pag.contracts.artifacts import PipelineArtifacts
from pag.contracts.schemas import RunConfig, SampleRecord

__all__ = ["run_pipeline"]


def run_pipeline(
    run_config: RunConfig,
    samples: list[SampleRecord] | tuple[SampleRecord, ...],
    *,
    persist: bool = False,
) -> PipelineArtifacts:
    from pag.orchestration.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(run_config, samples, persist=persist)
