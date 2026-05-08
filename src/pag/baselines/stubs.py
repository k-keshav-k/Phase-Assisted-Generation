from __future__ import annotations

from collections.abc import Sequence

from pag.contracts.artifacts import BaselineRunArtifacts
from pag.contracts.enums import StageName
from pag.contracts.schemas import (
    DecodingResult,
    GenerationRequest,
    GenerationTrace,
    RunConfig,
    RunSummary,
    SampleRecord,
    TokenSignal,
    TraceStep,
)
from pag.utils.ids import build_request_id


def mock_baseline_runner(
    run_config: RunConfig,
    samples: Sequence[SampleRecord],
) -> BaselineRunArtifacts:
    """Default deterministic baseline stub.

    TODO(team-baselines): Replace the mock request/trace/signal/completion generation with
    real model adapter and fixed-decoding logic while preserving the returned artifact shapes.
    """

    requests: list[GenerationRequest] = []
    traces: list[GenerationTrace] = []
    token_signals: list[TokenSignal] = []
    completions: list[DecodingResult] = []

    total_tokens = 0
    total_steps = 0
    for sample_index, sample in enumerate(samples):
        request_id = build_request_id(run_config.run_id, sample.sample_id)
        requests.append(
            GenerationRequest(
                run_id=run_config.run_id,
                request_id=request_id,
                sample=sample,
                model=run_config.model,
                decoding=run_config.decoding,
                requested_artifacts=["trace", "token_signals", "completion"],
            )
        )

        final_tokens = [
            "baseline",
            f"sample{sample_index + 1}",
            run_config.decoding.strategy,
            "completion",
        ]
        step_count = max(1, min(2, run_config.decoding.refinement_steps + 1))
        steps = [
            TraceStep(
                step_index=step_index,
                chunk_size=run_config.decoding.chunk_size,
                refinement_steps=run_config.decoding.refinement_steps,
                emitted_tokens=final_tokens[step_index::step_count],
                metadata={"source": "mock-baseline"},
            )
            for step_index in range(step_count)
        ]
        traces.append(
            GenerationTrace(
                sample_id=sample.sample_id,
                request_id=request_id,
                steps=steps,
                final_tokens=final_tokens,
                metadata={"decoder_mode": run_config.decoding.strategy},
            )
        )

        for token_index, token in enumerate(final_tokens):
            token_signals.append(
                TokenSignal(
                    sample_id=sample.sample_id,
                    token_index=token_index,
                    token_text=token,
                    step_index=token_index % step_count,
                    values={
                        "entropy": round(0.35 + (token_index * 0.1), 3),
                        "confidence": round(0.9 - (token_index * 0.08), 3),
                    },
                )
            )

        completion = " ".join(final_tokens)
        completions.append(
            DecodingResult(
                sample_id=sample.sample_id,
                request_id=request_id,
                completion=completion,
                tokens=final_tokens,
                token_ids=list(range(len(final_tokens))),
                metadata={"source": "mock-baseline"},
            )
        )
        total_tokens += len(final_tokens)
        total_steps += len(steps)

    summary = RunSummary(
        run_id=run_config.run_id,
        stage=StageName.BASELINE.value,
        num_samples=len(samples),
        metrics={
            "num_requests": float(len(requests)),
            "tokens_generated": float(total_tokens),
            "avg_trace_steps": float(total_steps / len(samples)) if samples else 0.0,
        },
        metadata={"implementation": "mock_baseline_runner"},
    )

    return BaselineRunArtifacts(
        run_config=run_config,
        samples=list(samples),
        requests=requests,
        traces=traces,
        token_signals=token_signals,
        completions=completions,
        summary=summary,
    )
