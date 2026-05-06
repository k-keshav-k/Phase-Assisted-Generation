"""Evaluate a trained phase_predict checkpoint with cost-aware metrics.

Reports prediction quality (top-1 acc, MAE) AND the headline metric — total
NFE per problem if the predictor's schedule had been used instead of
AdaBlock's. Negative ``delta_pct`` means the predictor would have used fewer
NFEs than AdaBlock.

Usage
-----
    python scripts/eval_phase_predict.py \\
        --checkpoint output/ablations/large_ws67_d256_h8_l1_dp0_lr1.0_bestval=0.095973.pt \\
        --jsonl      traces/rich/stab_tuples_conf_test_rich.jsonl

    # Rollout mode (use model's own past predictions, not ground truth):
    python scripts/eval_phase_predict.py \\
        --checkpoint <ckpt> --jsonl <file> --mode rollout

    # If the checkpoint was trained on extra input features, pass them with
    # the same order used at training time:
    python scripts/eval_phase_predict.py --checkpoint <ckpt> --jsonl <file> \\
        --features block_size nfe mean_top1_confidence min_top1_confidence \\
                  digit_fraction delimiter_fraction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from phase_predict.evaluate import evaluate, format_report
from phase_predict.predict import Predictor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cost-aware evaluation for phase_predict checkpoints."
    )
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to phase_tuples JSONL test file.")
    ap.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Ordered list of input feature field names. Defaults to the list saved "
            "in the checkpoint, falling back to [block_size, nfe]."
        ),
    )
    ap.add_argument(
        "--block-field",
        type=str,
        default="block_size",
        help="Name of the block-size field in the JSONL (default: block_size).",
    )
    ap.add_argument(
        "--nfe-field",
        type=str,
        default="nfe",
        help="Name of the NFE field in the JSONL (default: nfe).",
    )
    ap.add_argument(
        "--mode",
        choices=("teacher_forced", "rollout"),
        default="teacher_forced",
        help=(
            "teacher_forced: each prediction conditions on TRUE prior blocks. "
            "rollout: each prediction conditions on the model's own previous outputs."
        ),
    )
    ap.add_argument(
        "--min-history",
        type=int,
        default=1,
        help="Skip predicting the first N blocks of each problem (counted as warmup NFEs).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override (e.g. cuda, cpu). Defaults to cuda if available.",
    )
    ap.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="If set, write the EvalReport as JSON to this path.",
    )
    ap.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="Optional: write per-block predictions as JSONL for offline analysis.",
    )
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    predictor = Predictor.from_checkpoint(str(args.checkpoint), device=device)
    print(
        f"Loaded checkpoint: {args.checkpoint}\n"
        f"  device           = {device}\n"
        f"  input_tuple_size = {predictor.config.input_tuple_size}\n"
        f"  output_tuple_size= {predictor.config.output_tuple_size}\n"
        f"  window_size      = {predictor.config.window_size}\n"
        f"  input_fields     = {predictor.input_fields}"
    )

    report = evaluate(
        predictor,
        args.jsonl,
        feature_fields=args.features,
        output_fields=(args.block_field, args.nfe_field),
        mode=args.mode,
        min_history=args.min_history,
        dump_predictions_to=args.dump_predictions,
    )

    print()
    print(format_report(report))

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport written to {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
