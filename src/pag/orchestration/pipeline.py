from __future__ import annotations

from collections.abc import Sequence

from pag.baselines import run_baseline
from pag.contracts.artifacts import PipelineArtifacts
from pag.contracts.schemas import RunConfig, SampleRecord
from pag.evaluation import evaluate_runs
from pag.phases import run_phase_analysis
from pag.scheduler import run_adaptive_decoding
from pag.utils.io import persist_pipeline_artifacts


def run_pipeline(
    run_config: RunConfig,
    samples: Sequence[SampleRecord],
    *,
    persist: bool = False,
) -> PipelineArtifacts:
    baseline_artifacts = run_baseline(run_config, samples)
    phase_artifacts = run_phase_analysis(run_config, baseline_artifacts)
    adaptive_artifacts = run_adaptive_decoding(run_config, baseline_artifacts, phase_artifacts)
    evaluation_artifacts = evaluate_runs(run_config, baseline_artifacts, adaptive_artifacts)
    artifacts = PipelineArtifacts(
        run_config=run_config,
        baseline=baseline_artifacts,
        phases=phase_artifacts,
        adaptive=adaptive_artifacts,
        evaluation=evaluation_artifacts,
        metadata={"persisted": persist},
    )
    if persist:
        persist_pipeline_artifacts(artifacts)
    return artifacts
