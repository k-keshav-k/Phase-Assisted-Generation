from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pag.experiments.statistics import (
    correctness_matrix,
    exact_mcnemar,
    pair_records,
    paired_bootstrap,
    wilson_interval,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flat_rows(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        grade = record.get("grade", {})
        rows.append(
            {
                "sample_id": record["sample_id"],
                "method": record["method"],
                "is_correct": grade.get("is_correct") if isinstance(grade, dict) else None,
                "total_nfe": record.get("total_nfe"),
                "elapsed_sec": record.get("elapsed_sec"),
                "scheduler_predict_time_sec": record.get("scheduler_predict_time_sec", 0.0),
            }
        )
    return rows


def summarize_pair(
    candidate: list[dict[str, object]],
    baseline: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    pairs = pair_records(candidate, baseline)
    candidate_correct = [bool(left["grade"]["is_correct"]) for left, _ in pairs]
    baseline_correct = [bool(right["grade"]["is_correct"]) for _, right in pairs]
    candidate_nfe = [float(left["total_nfe"]) for left, _ in pairs]
    baseline_nfe = [float(right["total_nfe"]) for _, right in pairs]
    matrix = correctness_matrix(candidate_correct, baseline_correct)
    latency_left = [float(left["elapsed_sec"]) for left, _ in pairs]
    latency_right = [float(right["elapsed_sec"]) for _, right in pairs]
    return {
        "count": len(pairs),
        "candidate_accuracy": asdict(wilson_interval(sum(candidate_correct), len(pairs))),
        "baseline_accuracy": asdict(wilson_interval(sum(baseline_correct), len(pairs))),
        "accuracy_difference": asdict(
            paired_bootstrap(
                candidate_correct,
                baseline_correct,
                samples=bootstrap_samples,
                seed=seed,
            )
        ),
        "correctness_matrix": asdict(matrix),
        "mcnemar": asdict(exact_mcnemar(matrix)),
        "nfe_difference": asdict(
            paired_bootstrap(candidate_nfe, baseline_nfe, samples=bootstrap_samples, seed=seed)
        ),
        "latency_difference": asdict(
            paired_bootstrap(latency_left, latency_right, samples=bootstrap_samples, seed=seed)
        ),
    }


def write_report(
    output_dir: str | Path,
    *,
    stages: dict[str, dict[str, list[dict[str, object]]]],
    bootstrap_samples: int,
    seed: int,
    confirmatory_ids: set[str],
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for stage, methods in stages.items():
        rows = [record for records in methods.values() for record in records]
        _write_csv(output / f"{stage}_results.csv", _flat_rows(rows))
        if "adablock" not in methods:
            continue
        stage_summary: dict[str, object] = {}
        for method, records in methods.items():
            if method == "adablock":
                continue
            stage_summary[method] = summarize_pair(
                records,
                methods["adablock"],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            if stage == "gsm8k_test":
                candidate_clean = [row for row in records if row["sample_id"] in confirmatory_ids]
                baseline_clean = [
                    row for row in methods["adablock"] if row["sample_id"] in confirmatory_ids
                ]
                stage_summary[f"{method}_confirmatory"] = summarize_pair(
                    candidate_clean,
                    baseline_clean,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
        summary[stage] = stage_summary
    _write_json(output / "summary.json", summary)
    _write_json(output / "paired_statistics.json", summary)
    _write_latex_table(output / "tables" / "gsm8k_results.tex", summary.get("gsm8k_test", {}))
    _write_nfe_figure(output / "figures" / "nfe_deltas.pdf", stages.get("gsm8k_test", {}))
    return summary


def _write_latex_table(path: Path, rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = rows if isinstance(rows, dict) else {}
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Method & $\\Delta$NFE & $p_{McN}$ \\\\",
        "\\midrule",
    ]
    for name, payload in sorted(mapping.items()):
        if not isinstance(payload, dict) or name.endswith("_confirmatory"):
            continue
        safe_name = name.replace("_", "\\_")
        lines.append(
            f"{safe_name} & {payload['nfe_difference']['estimate']:.3f} & "
            f"{payload['mcnemar']['pvalue']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nfe_figure(path: Path, methods: dict[str, list[dict[str, object]]]) -> None:
    if "adablock" not in methods or "pag" not in methods:
        return
    pairs = pair_records(methods["pag"], methods["adablock"])
    deltas = [float(left["total_nfe"]) - float(right["total_nfe"]) for left, right in pairs]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    axis.hist(deltas, bins=30)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("PAG - AdaBlock total NFE")
    axis.set_ylabel("Prompts")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
