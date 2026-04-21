from __future__ import annotations

from phase_cpd.schema import TokenStepObservation, TraceRecord, TraceToken


def make_stabilized_trace(
    *,
    trace_id: str = "fixture-trace",
    prompt: str = "Explain fixture traces.",
    token_texts: list[str] | None = None,
) -> TraceRecord:
    final_tokens = token_texts or ["A", "B", "C", "D", "E", "F", "G"]
    tokens: list[TraceToken] = []
    cursor = 0
    for token_index, token_text in enumerate(final_tokens):
        stable_step = token_index + 1
        token_id = token_index + 10
        observations = [
            TokenStepObservation(
                step_index=0,
                token_id=99,
                token_text="<|mask|>",
                top1_prob=0.1,
                top2_prob=0.08,
                extras={"entropy": 2.0, "is_mask": 1.0, "changed_from_prev_step": 0.0},
            ),
            TokenStepObservation(
                step_index=stable_step,
                token_id=token_id,
                token_text=token_text,
                top1_prob=0.5,
                top2_prob=0.1,
                extras={"entropy": 0.7, "is_mask": 0.0, "changed_from_prev_step": 1.0},
            ),
            TokenStepObservation(
                step_index=stable_step + 1,
                token_id=token_id,
                token_text=token_text,
                top1_prob=0.8,
                top2_prob=0.05,
                extras={"entropy": 0.3, "is_mask": 0.0, "changed_from_prev_step": 0.0},
            ),
        ]
        tokens.append(
            TraceToken(
                token_index=token_index,
                token_text=token_text,
                char_start=cursor,
                char_end=cursor + len(token_text),
                observations=observations,
            )
        )
        cursor += len(token_text)

    return TraceRecord(
        trace_id=trace_id,
        backend="dream",
        model_name="dream-test",
        prompt=prompt,
        final_text="".join(final_tokens),
        tokens=tokens,
        decoding_metadata={
            "run_id": "fixture-run",
            "trace_profile": "entropy_stochastic",
            "alg": "entropy",
            "temperature": 0.0,
            "alg_temp": 0.1,
            "seed": 0,
        },
    )
