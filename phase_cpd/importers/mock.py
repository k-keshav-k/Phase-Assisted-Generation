from __future__ import annotations

from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceToken


def build_mock_trace_examples() -> list[TraceRecord]:
    return [_adaptive_trace(), _quality_trace()]


def _adaptive_trace() -> TraceRecord:
    return _build_trace(
        trace_id="mock-adaptive-001",
        prompt="Explain why adaptive diffusion decoding can help generation quality.",
        backend="mock",
        model_name="mock-diffusion-lm",
        tags=["mock", "adaptive", "quality"],
        run_id="mock-run-001",
        token_texts=[
            "Adaptive",
            " decoding",
            " allocates",
            " more",
            " refinement",
            " to",
            " hard",
            " regions",
            ".",
        ],
        top1_probs=[
            [0.55, 0.78, 0.91],
            [0.53, 0.75, 0.89],
            [0.50, 0.72, 0.87],
            [0.37, 0.46, 0.54],
            [0.34, 0.43, 0.50],
            [0.31, 0.40, 0.48],
            [0.49, 0.71, 0.85],
            [0.47, 0.68, 0.83],
            [0.46, 0.66, 0.82],
        ],
    )


def _quality_trace() -> TraceRecord:
    return _build_trace(
        trace_id="mock-throughput-002",
        prompt="Summarize the tradeoff between throughput and quality in iterative decoding.",
        backend="mock",
        model_name="mock-diffusion-lm",
        tags=["mock", "throughput", "quality"],
        run_id="mock-run-002",
        token_texts=[
            "Quality",
            " improvements",
            " often",
            " require",
            " extra",
            " refinement",
            " and",
            " slower",
            " throughput",
            ".",
        ],
        top1_probs=[
            [0.59, 0.82, 0.94],
            [0.54, 0.76, 0.90],
            [0.52, 0.73, 0.87],
            [0.41, 0.55, 0.66],
            [0.39, 0.51, 0.63],
            [0.37, 0.48, 0.61],
            [0.34, 0.46, 0.58],
            [0.43, 0.61, 0.72],
            [0.48, 0.69, 0.78],
            [0.50, 0.72, 0.80],
        ],
    )


def _build_trace(
    *,
    trace_id: str,
    prompt: str,
    backend: str,
    model_name: str,
    tags: list[str],
    run_id: str,
    token_texts: list[str],
    top1_probs: list[list[float]],
) -> TraceRecord:
    tokens: list[TraceToken] = []
    offset = 0
    token_rows = zip(token_texts, top1_probs, strict=True)
    for index, (token_text, token_probabilities) in enumerate(token_rows):
        observations = [
            TokenStepObservation(
                step_index=step_index,
                top1_prob=probability,
                selected_logit=round(probability * 4.0, 4),
                top2_prob=round(max(probability - 0.12, 0.01), 4),
                extras={"stability": round((step_index + 1) / len(token_probabilities), 4)},
            )
            for step_index, probability in enumerate(token_probabilities)
        ]
        token_length = len(token_text)
        tokens.append(
            TraceToken(
                token_index=index,
                token_text=token_text,
                char_start=offset,
                char_end=offset + token_length,
                observations=observations,
            )
        )
        offset += token_length

    final_text = "".join(token.token_text for token in tokens)
    return TraceRecord(
        trace_id=trace_id,
        backend=backend,
        model_name=model_name,
        prompt=prompt,
        final_text=final_text,
        tokens=tokens,
        decoding_metadata={
            "run_id": run_id,
            "chunk_size": 4,
            "refinement_steps": 3,
            "temperature": 0.0,
            "sampler": "mock-refinement",
        },
        tags=tags,
        created_at="2026-04-18T00:00:00Z",
    )
