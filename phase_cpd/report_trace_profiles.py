from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_cpd.cpd import CPDParameters
from phase_cpd.importers.common import load_step_dump_as_trace
from phase_cpd.scheduler_dataset import SchedulerDatasetConfig, build_profile_report


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    source = Path(args.source)
    traces = _load_traces(source, args.glob)
    config = SchedulerDatasetConfig(
        feature_name=args.feature,
        detector_name=args.detector,
        kernel=args.kernel,
        cpd_params=CPDParameters(
            cost=args.cost,
            penalty=args.penalty,
            min_segment_length=args.min_segment_length,
            smoothing_window=args.smoothing_window,
        ),
    )
    report = build_profile_report(traces, config=config)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(output_path)
        return 0

    print(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize scheduler-supervision quality by Dream trace profile."
    )
    parser.add_argument("--source", required=True, help="Trace JSON file or directory.")
    parser.add_argument("--output", help="Optional JSON report output path.")
    parser.add_argument("--glob", default="*.json", help="Glob used when --source is a directory.")
    parser.add_argument("--feature", default="stabilizing_entropy")
    parser.add_argument("--detector", default="pelt")
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--cost", default="l2")
    parser.add_argument("--penalty", type=float, default=0.1)
    parser.add_argument("--min-segment-length", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=1)
    return parser


def _load_traces(source: Path, pattern: str):
    paths = sorted(source.glob(pattern)) if source.is_dir() else [source]
    if not paths:
        msg = f"No trace files matched in {source}"
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
