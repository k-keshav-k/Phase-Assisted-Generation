from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phase_cpd.io import load_trace
from phase_cpd.schema import TraceRecord


@dataclass(slots=True)
class CatalogEntry:
    trace_id: str
    path: Path
    backend: str
    model_name: str
    prompt_preview: str
    tags: tuple[str, ...]
    run_id: str | None
    label: str


def default_trace_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "traces_real"


def list_catalog_entries(trace_dir: str | Path | None = None) -> list[CatalogEntry]:
    directory = Path(trace_dir) if trace_dir is not None else default_trace_dir()
    if not directory.exists():
        msg = f"Trace directory does not exist: {directory}"
        raise FileNotFoundError(msg)

    paths = sorted(directory.glob("*.json"))
    if not paths:
        msg = f"No trace JSON files were found in {directory}"
        raise FileNotFoundError(msg)

    entries: list[CatalogEntry] = []
    for path in paths:
        trace = load_trace(path)
        run_id = _extract_run_id(trace)
        entries.append(
            CatalogEntry(
                trace_id=trace.trace_id,
                path=path,
                backend=trace.backend,
                model_name=trace.model_name,
                prompt_preview=_prompt_preview(trace.prompt),
                tags=tuple(trace.tags),
                run_id=run_id,
                label=_build_label(trace, run_id),
            )
        )
    return entries


def filter_catalog_entries(
    entries: list[CatalogEntry],
    *,
    backend: str | None = None,
    model_name: str | None = None,
    required_tags: set[str] | None = None,
    run_id: str | None = None,
) -> list[CatalogEntry]:
    required_tags = required_tags or set()
    filtered: list[CatalogEntry] = []
    for entry in entries:
        if backend and entry.backend != backend:
            continue
        if model_name and entry.model_name != model_name:
            continue
        if run_id and entry.run_id != run_id:
            continue
        if required_tags and not required_tags.issubset(set(entry.tags)):
            continue
        filtered.append(entry)
    return filtered


def load_trace_by_id(trace_id: str, trace_dir: str | Path | None = None) -> TraceRecord:
    for entry in list_catalog_entries(trace_dir):
        if entry.trace_id == trace_id:
            return load_trace(entry.path)
    msg = f"Unknown trace_id: {trace_id}"
    raise KeyError(msg)


def _build_label(trace: TraceRecord, run_id: str | None) -> str:
    run_suffix = f" | {run_id}" if run_id else ""
    prompt_preview = _prompt_preview(trace.prompt)
    return f"[{trace.backend}] {trace.model_name}{run_suffix} | {trace.trace_id} | {prompt_preview}"


def _prompt_preview(prompt: str, limit: int = 72) -> str:
    prompt = " ".join(prompt.split())
    if len(prompt) <= limit:
        return prompt
    return f"{prompt[: limit - 3]}..."


def _extract_run_id(trace: TraceRecord) -> str | None:
    value = trace.decoding_metadata.get("run_id")
    if value is None:
        return None
    return str(value)
