"""Utilities to extract PhaseTuple sequences from phase_cpd trace data.

This module bridges the phase_cpd data format and the phase_predict model
input format.  It can be imported independently of the rest of the PAG
pipeline.

The extraction strategy is:
  - For each token in a TraceRecord, compute:
      * refinement_steps:  the total number of diffusion steps observed for
        the token (i.e. len(token.observations)).
  - CPD (change-point detection) segments the trace into blocks.  The
    block_size is the number of tokens in each CPD segment.
  - Per-segment statistics (mean refinement_steps, rounded to int) become
    one PhaseTuple per segment.

If no CPD segmentation is desired, ``extract_per_token`` returns one
PhaseTuple per token (block_size=1).

Additional fields (e.g. stabilizing_steps) can be added to PhaseTuple and
to the extraction functions below without changing the model architecture —
just update ``ModelConfig.tuple_size`` to match.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phase_predict.schema import PhaseTuple


def _stabilizing_step(observations: list) -> int:
    """Return the earliest step index at which the token identity stabilised.

    A token is considered stable at step *s* when its identity at step *s*
    matches the final identity **and** all subsequent steps also match.

    Args:
        observations: list of ``TokenStepObservation`` objects from a
                      ``TraceToken``, already sorted by ``step_index``.

    Returns:
        The stabilising step index, or 0 if observations are empty or
        identity history is unavailable (graceful fallback).
    """
    if not observations:
        return 0

    sorted_obs = sorted(observations, key=lambda o: o.step_index)

    # determine final identity
    last = sorted_obs[-1]
    final_id = last.token_id if last.token_id is not None else last.token_text
    if final_id is None:
        return 0

    # walk forward to find the first step from which identity never changes
    for i, obs in enumerate(sorted_obs):
        candidate = obs.token_id if obs.token_id is not None else obs.token_text
        if candidate != final_id:
            continue
        if all(
            (o.token_id if o.token_id is not None else o.token_text) == final_id
            for o in sorted_obs[i:]
        ):
            return int(obs.step_index)
    return 0


def extract_per_token(trace: object) -> list[PhaseTuple]:
    """Extract one :class:`~phase_predict.schema.PhaseTuple` per token.

    Args:
        trace: a ``phase_cpd.schema.TraceRecord`` instance.

    Returns:
        Ordered list of PhaseTuples, one per token in ``trace.tokens``.
        ``block_size`` is always 1 in this representation.
    """
    tuples: list[PhaseTuple] = []
    for token in trace.tokens:  # type: ignore[union-attr]
        ref = len(token.observations)
        tuples.append(PhaseTuple(block_size=1, refinement_steps=ref))
    return tuples


def extract_per_segment(
    trace: object,
    breakpoints: Sequence[int],
) -> list[PhaseTuple]:
    """Extract one :class:`~phase_predict.schema.PhaseTuple` per CPD segment.

    Args:
        trace:       a ``phase_cpd.schema.TraceRecord`` instance.
        breakpoints: list of token boundary indices from a CPD detector
                     (see ``phase_cpd.cpd``).  May be empty, in which case
                     the whole trace is treated as a single segment.

    Returns:
        One PhaseTuple per segment.  ``block_size`` equals the number of
        tokens in the segment; ``refinement_steps`` is the rounded mean
        across tokens in the segment.
    """
    tokens = trace.tokens  # type: ignore[union-attr]
    n = len(tokens)
    if n == 0:
        return []

    boundaries = [0, *sorted(int(b) for b in breakpoints if 0 < int(b) < n), n]
    tuples: list[PhaseTuple] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_tokens = tokens[start:end]
        block_size = end - start
        ref_vals = [len(t.observations) for t in segment_tokens]
        mean_ref = round(sum(ref_vals) / len(ref_vals)) if ref_vals else 0
        tuples.append(
            PhaseTuple(
                block_size=block_size,
                refinement_steps=mean_ref,
            )
        )
    return tuples


def tuples_from_trace(
    trace: object,
    *,
    breakpoints: Sequence[int] | None = None,
) -> list[PhaseTuple]:
    """Convenience function: extract PhaseTuples from a TraceRecord.

    When *breakpoints* are provided the trace is segmented and one tuple
    per segment is returned.  Otherwise one tuple per token is returned.

    Args:
        trace:       a ``phase_cpd.schema.TraceRecord``.
        breakpoints: optional CPD breakpoint indices.

    Returns:
        Ordered list of :class:`~phase_predict.schema.PhaseTuple` values.
    """
    if breakpoints is not None:
        return extract_per_segment(trace, breakpoints)
    return extract_per_token(trace)


def tuples_from_token_summaries(
    token_summaries: Sequence[dict[str, Any]],
    *,
    first_field: str = "tau_commit",
    second_field: str = "tau_stable",
    default_value: int = 0,
) -> list[PhaseTuple]:
    """Convert token summary dictionaries to PhaseTuple values.

    This is intended for JSONL traces containing
    ``decoding_metadata.token_summaries`` where each item may include
    fields such as ``tau_commit`` and ``tau_stable``.

    Args:
        token_summaries: sequence of per-token summary dictionaries.
        first_field: key used as the first tuple component.
        second_field: key used as the second tuple component.
        default_value: fallback integer when a key is missing or null.

    Returns:
        A list of :class:`~phase_predict.schema.PhaseTuple` values.
    """
    tuples: list[PhaseTuple] = []
    for summary in token_summaries:
        first_raw = summary.get(first_field, default_value)
        second_raw = summary.get(second_field, default_value)

        first_value = default_value if first_raw is None else int(first_raw)
        second_value = default_value if second_raw is None else int(second_raw)

        tuples.append(
            PhaseTuple(
                block_size=max(0, first_value),
                refinement_steps=max(0, second_value),
            )
        )
    return tuples


def tuple_sequences_from_trace_jsonl(
    jsonl_path: str | Path,
    *,
    first_field: str = "tau_commit",
    second_field: str = "tau_stable",
) -> list[list[PhaseTuple]]:
    """Load tuple sequences from a trace JSONL file.

    Each JSON line is expected to include
    ``decoding_metadata.token_summaries``.

    Args:
        jsonl_path: path to JSONL trace file.
        first_field: key used as the first tuple component.
        second_field: key used as the second tuple component.

    Returns:
        A list of tuple sequences, one sequence per JSONL record.
    """
    path = Path(jsonl_path)
    sequences: list[list[PhaseTuple]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            metadata = record.get("decoding_metadata", {})
            summaries = metadata.get("token_summaries", [])
            if not isinstance(summaries, list):
                continue

            seq = tuples_from_token_summaries(
                summaries,
                first_field=first_field,
                second_field=second_field,
            )
            if seq:
                sequences.append(seq)

    return sequences
