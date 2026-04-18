from __future__ import annotations

import argparse
from pathlib import Path

from phase_cpd.importers.dream import import_dream_trace
from phase_cpd.importers.llada import import_llada_trace
from phase_cpd.importers.mock import build_mock_trace_examples
from phase_cpd.io import save_trace
from phase_cpd.schema import TraceRecord


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "mock":
        traces = build_mock_trace_examples()
    else:
        if not args.source:
            parser.error("--source is required for dream and llada backends")
        traces = _load_real_backend_traces(args.backend, Path(args.source), args.glob)

    saved_paths = []
    for trace in traces:
        target = output_dir / f"{trace.trace_id}.json"
        saved_paths.append(save_trace(target, trace))

    for path in saved_paths:
        print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw trace dumps into phase_cpd TraceRecord JSON files."
    )
    parser.add_argument(
        "--backend",
        choices=["dream", "llada", "mock"],
        required=True,
        help="Backend trace format to convert.",
    )
    parser.add_argument(
        "--source",
        help="Raw trace JSON file or directory. Not required for the mock backend.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where unified phase_cpd trace JSON files will be written.",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob used when --source points to a directory.",
    )
    return parser


def _load_real_backend_traces(backend: str, source: Path, pattern: str) -> list[TraceRecord]:
    if source.is_dir():
        paths = sorted(source.glob(pattern))
    else:
        paths = [source]

    if not paths:
        msg = f"No raw trace files matched in {source}"
        raise FileNotFoundError(msg)

    importer = import_dream_trace if backend == "dream" else import_llada_trace
    return [importer(path) for path in paths]


if __name__ == "__main__":
    raise SystemExit(main())
