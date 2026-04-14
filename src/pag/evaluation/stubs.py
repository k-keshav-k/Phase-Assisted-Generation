from __future__ import annotations

from pag.contracts.artifacts import AdaptiveRunArtifacts, BaselineRunArtifacts, EvaluationArtifacts
from pag.contracts.enums import StageName
from pag.contracts.schemas import EvaluationRecord, RunConfig, RunSummary


def mock_evaluator(
    run_config: RunConfig,
    baseline_artifacts: BaselineRunArtifacts,
    adaptive_artifacts: AdaptiveRunArtifacts,
) -> EvaluationArtifacts:
    records: list[EvaluationRecord] = []
    adaptive_by_sample = {
        result.sample_id: result for result in adaptive_artifacts.adaptive_results
    }
    for completion in baseline_artifacts.completions:
        adaptive_result = adaptive_by_sample[completion.sample_id]
        records.append(
            EvaluationRecord(
                sample_id=completion.sample_id,
                baseline_completion=completion.completion,
                adaptive_completion=adaptive_result.completion,
                metrics={
                    "baseline_length": float(len(completion.tokens)),
                    "adaptive_length": float(len(adaptive_result.tokens)),
                },
                notes={"implementation": "mock_evaluator"},
            )
        )

    summary = RunSummary(
        run_id=run_config.run_id,
        stage=StageName.EVALUATION.value,
        num_samples=len(records),
        metrics={"num_records": float(len(records))},
        metadata={"implementation": "mock_evaluator"},
    )
    return EvaluationArtifacts(run_config=run_config, records=records, summary=summary)

