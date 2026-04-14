from __future__ import annotations

from collections import defaultdict

from pag.contracts.artifacts import BaselineRunArtifacts, PhaseArtifacts
from pag.contracts.enums import PhaseLabel, StageName
from pag.contracts.schemas import (
    PhasePrediction,
    PhaseSpan,
    PredictorDatasetItem,
    RunConfig,
    RunSummary,
)


def mock_phase_runner(
    run_config: RunConfig,
    baseline_artifacts: BaselineRunArtifacts,
    hidden_state_features: dict[str, list[dict[str, float]]] | None = None,
) -> PhaseArtifacts:
    """Default deterministic phase-analysis stub.

    TODO(team-phases): Replace the mock annotations, dataset assembly, and predictions with
    real signal extraction and phase-prediction logic while preserving returned artifact shapes.
    """

    hidden_state_features = hidden_state_features or {}
    grouped_signals = defaultdict(list)
    for signal in baseline_artifacts.token_signals:
        grouped_signals[signal.sample_id].append(signal)

    annotations: list[PhaseSpan] = []
    predictor_dataset: list[PredictorDatasetItem] = []
    predictions: list[PhasePrediction] = []

    for trace in baseline_artifacts.traces:
        sample_signals = sorted(grouped_signals[trace.sample_id], key=lambda item: item.token_index)
        midpoint = max(0, len(trace.final_tokens) // 2)
        easy_span = PhaseSpan(
            sample_id=trace.sample_id,
            start_token=0,
            end_token=max(0, midpoint - 1),
            label=PhaseLabel.EASY.value,
            score=0.75,
            metadata={"source": "mock-annotator"},
        )
        hard_span = PhaseSpan(
            sample_id=trace.sample_id,
            start_token=midpoint,
            end_token=max(midpoint, len(trace.final_tokens) - 1),
            label=PhaseLabel.HARD.value,
            score=0.68,
            metadata={"source": "mock-annotator"},
        )
        sample_spans = [easy_span, hard_span]
        annotations.extend(sample_spans)

        for signal in sample_signals:
            label = (
                PhaseLabel.EASY.value
                if signal.token_index < midpoint
                else PhaseLabel.HARD.value
            )
            predictor_dataset.append(
                PredictorDatasetItem(
                    sample_id=trace.sample_id,
                    token_index=signal.token_index,
                    features={
                        "entropy": signal.values.get("entropy", 0.0),
                        "confidence": signal.values.get("confidence", 0.0),
                        "hidden_feature_count": float(
                            len(hidden_state_features.get(trace.sample_id, []))
                        ),
                    },
                    label=label,
                    metadata={"token_text": signal.token_text},
                )
            )

        predictions.append(
            PhasePrediction(
                sample_id=trace.sample_id,
                predictor_name=run_config.predictor.name,
                spans=sample_spans,
                features={"num_trace_steps": float(len(trace.steps))},
                metadata={"implementation": "mock_phase_runner"},
            )
        )

    summary = RunSummary(
        run_id=run_config.run_id,
        stage=StageName.PHASES.value,
        num_samples=len(baseline_artifacts.samples),
        metrics={
            "num_annotations": float(len(annotations)),
            "num_predictions": float(len(predictions)),
            "num_predictor_items": float(len(predictor_dataset)),
        },
        metadata={"implementation": "mock_phase_runner"},
    )

    return PhaseArtifacts(
        run_config=run_config,
        phase_annotations=annotations,
        predictor_dataset=predictor_dataset,
        predictions=predictions,
        predictor_metadata={
            "predictor_name": run_config.predictor.name,
            "label_space": run_config.predictor.label_space,
            "uses_hidden_state_features": bool(hidden_state_features),
        },
        summary=summary,
    )
