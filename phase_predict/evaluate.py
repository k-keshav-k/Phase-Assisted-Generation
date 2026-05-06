"""Cost-aware evaluator for phase_predict checkpoints.

Compares a trained Predictor against the AdaBlock baseline schedule recorded
in the same JSONL traces. Reports both prediction quality and the metric we
actually care about: total NFE per problem if we'd used the predictor's
schedule instead of AdaBlock's.

Two evaluation modes are supported:

  ``teacher_forced`` (default)
      For each block ``i`` in a problem, build the context from the *true*
      previous blocks ``0..i-1`` and predict block ``i``. This isolates the
      one-step prediction error.

  ``rollout``
      Walk through the problem left-to-right. For block ``i`` use the
      previously *predicted* output (block_size, refinement_steps) rather
      than the ground truth. Other input features beyond the two outputs
      are still taken from the trace (we don't have a generative model for
      them). This is the more realistic deployment metric.

Headline numbers reported:

  * ``block_size``: top-1 accuracy, top-3 accuracy, MAE, ±1 accuracy.
  * ``nfe``: MAE, RMSE, fraction within ±1, fraction under-allocated.
  * Per-problem ``predicted_total_nfe`` vs ``actual_total_nfe`` (the AdaBlock
    schedule). ``nfe_delta_pct`` < 0 means the predictor would have used
    fewer NFEs than AdaBlock.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phase_predict.predict import Predictor
from phase_predict.schema import ExtendedPhaseTuple, PhaseTuple


@dataclass
class BlockPrediction:
    """One per-block prediction record (for diagnostics / dumping)."""

    sample_id: str
    block_index: int
    actual_block_size: int
    actual_nfe: int
    predicted_block_size: int
    predicted_nfe: int


@dataclass
class EvalReport:
    """Aggregated evaluation results."""

    mode: str
    n_problems: int
    n_blocks: int

    # block_size metrics
    block_top1_acc: float = 0.0
    block_top1_acc_within_1: float = 0.0
    block_mae: float = 0.0
    block_rmse: float = 0.0

    # nfe metrics
    nfe_mae: float = 0.0
    nfe_rmse: float = 0.0
    nfe_within_1: float = 0.0
    nfe_under_allocated_frac: float = 0.0

    # per-problem total NFE comparison (predictor vs adablock baseline)
    actual_total_nfe_mean: float = 0.0
    predicted_total_nfe_mean: float = 0.0
    nfe_delta_pct_mean: float = 0.0  # negative => predictor uses fewer NFEs
    problems_under_baseline_frac: float = 0.0

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_problems": self.n_problems,
            "n_blocks": self.n_blocks,
            "block_size": {
                "top1_acc": self.block_top1_acc,
                "within_1_acc": self.block_top1_acc_within_1,
                "mae": self.block_mae,
                "rmse": self.block_rmse,
            },
            "nfe": {
                "mae": self.nfe_mae,
                "rmse": self.nfe_rmse,
                "within_1": self.nfe_within_1,
                "under_allocated_frac": self.nfe_under_allocated_frac,
            },
            "total_nfe_per_problem": {
                "actual_mean": self.actual_total_nfe_mean,
                "predicted_mean": self.predicted_total_nfe_mean,
                "delta_pct_mean": self.nfe_delta_pct_mean,
                "problems_under_baseline_frac": self.problems_under_baseline_frac,
            },
            "extra": self.extra,
        }


def _load_problems(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tuples = rec.get("tuples")
            if isinstance(tuples, list) and tuples:
                out.append(rec)
    return out


def _build_extended_context(
    problem_tuples: list[dict],
    upto_excl: int,
    feature_fields: list[str],
) -> list[ExtendedPhaseTuple]:
    return [
        ExtendedPhaseTuple(values={f: int(round(float(t.get(f, 0) or 0))) for f in feature_fields})
        for t in problem_tuples[:upto_excl]
    ]


def _override_block_in_context(
    ctx: list[ExtendedPhaseTuple],
    last_idx: int,
    output_fields: tuple[str, str],
    pred_block: int,
    pred_nfe: int,
) -> None:
    """Replace the last context entry's output fields with the model's prediction.

    Used by rollout mode so the next prediction sees the model's own past
    block_size/nfe rather than the ground truth.
    """
    if last_idx < 0 or last_idx >= len(ctx):
        return
    bf, nf = output_fields
    ctx[last_idx].values[bf] = int(pred_block)
    ctx[last_idx].values[nf] = int(pred_nfe)


def _aggregate(
    *,
    mode: str,
    blocks: list[BlockPrediction],
    per_problem_actual_nfe: list[int],
    per_problem_predicted_nfe: list[int],
) -> EvalReport:
    n_blocks = len(blocks)
    n_problems = len(per_problem_actual_nfe)

    if n_blocks == 0:
        return EvalReport(mode=mode, n_problems=n_problems, n_blocks=0)

    # block_size metrics
    bs_correct = sum(1 for b in blocks if b.predicted_block_size == b.actual_block_size)
    bs_within_1 = sum(
        1 for b in blocks if abs(b.predicted_block_size - b.actual_block_size) <= 1
    )
    bs_abs_err = [abs(b.predicted_block_size - b.actual_block_size) for b in blocks]
    bs_sq_err = [(b.predicted_block_size - b.actual_block_size) ** 2 for b in blocks]

    # nfe metrics
    nfe_abs_err = [abs(b.predicted_nfe - b.actual_nfe) for b in blocks]
    nfe_sq_err = [(b.predicted_nfe - b.actual_nfe) ** 2 for b in blocks]
    nfe_within_1 = sum(1 for b in blocks if abs(b.predicted_nfe - b.actual_nfe) <= 1)
    nfe_under = sum(1 for b in blocks if b.predicted_nfe < b.actual_nfe)

    # per-problem totals (compute on what we actually ran — may exclude the
    # very first block if the model can't predict from an empty history)
    actual_mean = sum(per_problem_actual_nfe) / max(n_problems, 1)
    pred_mean = sum(per_problem_predicted_nfe) / max(n_problems, 1)
    deltas_pct: list[float] = []
    under_baseline = 0
    for actual, pred in zip(per_problem_actual_nfe, per_problem_predicted_nfe):
        if actual > 0:
            deltas_pct.append(100.0 * (pred - actual) / actual)
        if pred < actual:
            under_baseline += 1

    return EvalReport(
        mode=mode,
        n_problems=n_problems,
        n_blocks=n_blocks,
        block_top1_acc=bs_correct / n_blocks,
        block_top1_acc_within_1=bs_within_1 / n_blocks,
        block_mae=sum(bs_abs_err) / n_blocks,
        block_rmse=math.sqrt(sum(bs_sq_err) / n_blocks),
        nfe_mae=sum(nfe_abs_err) / n_blocks,
        nfe_rmse=math.sqrt(sum(nfe_sq_err) / n_blocks),
        nfe_within_1=nfe_within_1 / n_blocks,
        nfe_under_allocated_frac=nfe_under / n_blocks,
        actual_total_nfe_mean=actual_mean,
        predicted_total_nfe_mean=pred_mean,
        nfe_delta_pct_mean=(sum(deltas_pct) / len(deltas_pct)) if deltas_pct else 0.0,
        problems_under_baseline_frac=under_baseline / max(n_problems, 1),
    )


def evaluate(
    predictor: Predictor,
    jsonl_path: str | Path,
    *,
    feature_fields: list[str] | None = None,
    output_fields: tuple[str, str] = ("block_size", "nfe"),
    mode: str = "teacher_forced",
    min_history: int = 1,
    dump_predictions_to: str | Path | None = None,
) -> EvalReport:
    """Evaluate a Predictor against the AdaBlock-recorded schedule.

    Args:
        predictor:        a loaded :class:`Predictor`.
        jsonl_path:       path to a phase_tuples JSONL (e.g. the rich test set).
        feature_fields:   ordered list of input feature field names. If None,
                          defaults to ``predictor.input_fields`` if available,
                          otherwise the two output fields only.
        output_fields:    ``(block_field, nfe_field)`` from the JSONL.
        mode:             ``"teacher_forced"`` or ``"rollout"``.
        min_history:      minimum number of past blocks before we issue a
                          prediction (we still count NFEs for skipped blocks
                          using the *actual* values so per-problem totals are
                          comparable to AdaBlock).
        dump_predictions_to: optional path; if given, every per-block
                          prediction record is written here as JSONL.

    Returns:
        :class:`EvalReport` with all metrics.
    """
    if mode not in {"teacher_forced", "rollout"}:
        msg = f"Unknown evaluation mode: {mode}"
        raise ValueError(msg)

    path = Path(jsonl_path)
    problems = _load_problems(path)
    if not problems:
        msg = f"No problems loaded from {path}"
        raise ValueError(msg)

    if feature_fields is None:
        feature_fields = list(predictor.input_fields) if predictor.input_fields else list(output_fields)

    blocks: list[BlockPrediction] = []
    per_problem_actual_nfe: list[int] = []
    per_problem_predicted_nfe: list[int] = []

    dump_handle = None
    if dump_predictions_to is not None:
        dump_path = Path(dump_predictions_to)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_handle = dump_path.open("w", encoding="utf-8")

    bf, nf = output_fields

    try:
        for problem in problems:
            problem_tuples = problem["tuples"]
            sample_id = str(problem.get("sample_id", "?"))
            n = len(problem_tuples)
            if n <= min_history:
                continue

            ctx: list[ExtendedPhaseTuple] = _build_extended_context(
                problem_tuples, upto_excl=min_history, feature_fields=feature_fields
            )

            actual_total = sum(int(t.get(nf, 0) or 0) for t in problem_tuples)
            # for the first ``min_history`` blocks we do not predict; charge the
            # actual NFE of those blocks toward the predictor as well, since
            # they're a fixed warm-up cost shared with AdaBlock.
            warmup_nfe = sum(int(t.get(nf, 0) or 0) for t in problem_tuples[:min_history])
            predicted_total = warmup_nfe

            for tgt in range(min_history, n):
                actual_block = int(problem_tuples[tgt].get(bf, 0) or 0)
                actual_nfe_value = int(problem_tuples[tgt].get(nf, 0) or 0)

                result = predictor.predict(ctx)
                pred_block = int(result.predicted_tuple.block_size)
                pred_nfe = int(result.predicted_tuple.refinement_steps)

                blocks.append(
                    BlockPrediction(
                        sample_id=sample_id,
                        block_index=tgt,
                        actual_block_size=actual_block,
                        actual_nfe=actual_nfe_value,
                        predicted_block_size=pred_block,
                        predicted_nfe=pred_nfe,
                    )
                )
                if dump_handle is not None:
                    dump_handle.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "block_index": tgt,
                                "actual_block_size": actual_block,
                                "actual_nfe": actual_nfe_value,
                                "predicted_block_size": pred_block,
                                "predicted_nfe": pred_nfe,
                            }
                        )
                        + "\n"
                    )
                predicted_total += pred_nfe

                # extend context for the next iteration. In teacher_forced mode
                # we append the ground-truth tuple; in rollout mode we replace
                # the model's predicted output fields into the next context
                # entry while keeping the other features from the trace (we
                # don't have a generative model for them).
                next_entry = ExtendedPhaseTuple(
                    values={
                        f: int(round(float(problem_tuples[tgt].get(f, 0) or 0)))
                        for f in feature_fields
                    }
                )
                if mode == "rollout":
                    next_entry.values[bf] = pred_block
                    next_entry.values[nf] = pred_nfe
                ctx.append(next_entry)

            per_problem_actual_nfe.append(actual_total)
            per_problem_predicted_nfe.append(predicted_total)
    finally:
        if dump_handle is not None:
            dump_handle.close()

    return _aggregate(
        mode=mode,
        blocks=blocks,
        per_problem_actual_nfe=per_problem_actual_nfe,
        per_problem_predicted_nfe=per_problem_predicted_nfe,
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"Eval mode      : {report.mode}",
        f"Problems       : {report.n_problems}",
        f"Blocks scored  : {report.n_blocks}",
        "",
        "block_size",
        f"  top-1 acc    : {report.block_top1_acc:6.3f}",
        f"  within ±1    : {report.block_top1_acc_within_1:6.3f}",
        f"  MAE          : {report.block_mae:6.3f}",
        f"  RMSE         : {report.block_rmse:6.3f}",
        "",
        "nfe (refinement_steps)",
        f"  MAE          : {report.nfe_mae:6.3f}",
        f"  RMSE         : {report.nfe_rmse:6.3f}",
        f"  within ±1    : {report.nfe_within_1:6.3f}",
        f"  under-alloc  : {report.nfe_under_allocated_frac:6.3f}  (lower = safer)",
        "",
        "Per-problem total NFE (predictor schedule vs AdaBlock baseline)",
        f"  actual mean  : {report.actual_total_nfe_mean:8.2f}",
        f"  pred   mean  : {report.predicted_total_nfe_mean:8.2f}",
        f"  delta %      : {report.nfe_delta_pct_mean:+8.2f}%   (negative = fewer NFEs than AdaBlock)",
        f"  under-base.  : {report.problems_under_baseline_frac:6.3f}  (fraction of problems where pred < actual)",
    ]
    return "\n".join(lines)
