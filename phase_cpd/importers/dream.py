from __future__ import annotations

from pathlib import Path

from phase_cpd.importers.common import load_step_dump_as_trace
from phase_cpd.schema import TraceRecord


def import_dream_trace(source: str | Path) -> TraceRecord:
    """Import a Dream raw step-dump JSON into the unified phase_cpd trace schema.

    TODO: instrument the local Dream generation loop to write per-step token rows in the
    step-dump format described in phase_cpd/README.md. This importer intentionally stays thin:
    it assumes the raw artifact already contains final-token order plus per-step selected-token
    probabilities or logits.
    """

    source_path = Path(source)
    return load_step_dump_as_trace(
        source_path,
        backend="dream",
        default_model_name="dream-7b",
    )
