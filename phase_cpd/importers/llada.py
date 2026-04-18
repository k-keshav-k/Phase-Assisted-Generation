from __future__ import annotations

from pathlib import Path

from phase_cpd.importers.common import load_step_dump_as_trace
from phase_cpd.schema import TraceRecord


def import_llada_trace(source: str | Path) -> TraceRecord:
    """Import a LLaDA raw step-dump JSON into the unified phase_cpd trace schema.

    TODO: instrument the local LLaDA decoding loop so each step writes token rows with at least
    token text and either top1 probability or selected logit. This importer converts that raw
    dump into the offline phase_cpd format used by the Streamlit UI.
    """

    source_path = Path(source)
    return load_step_dump_as_trace(
        source_path,
        backend="llada",
        default_model_name="llada",
    )
