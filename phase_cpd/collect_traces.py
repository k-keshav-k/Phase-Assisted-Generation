from __future__ import annotations

import argparse
from pathlib import Path

from phase_cpd.importers.common import load_step_dump_as_trace
from phase_cpd.io import save_trace
from phase_cpd.schema import TraceRecord


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.source:
        parser.error("--source is required for the dream backend")
    traces = _load_dream_traces(Path(args.source), args.glob)

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
        choices=["dream"],
        required=True,
        help="Backend trace format to convert.",
    )
    parser.add_argument(
        "--source",
        help="Raw Dream trace JSON file or directory.",
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


def _load_dream_traces(source: Path, pattern: str) -> list[TraceRecord]:
    if source.is_dir():
        paths = sorted(source.glob(pattern))
    else:
        paths = [source]

    if not paths:
        msg = f"No raw trace files matched in {source}"
        raise FileNotFoundError(msg)

    return [
        load_step_dump_as_trace(
            path,
            backend="dream",
            default_model_name="dream-7b",
        )
        for path in paths
    ]


if __name__ == "__main__":
    raise SystemExit(main())
