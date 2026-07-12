from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from pag.experiments.claim_audit import ClaimThresholds, audit_claims
from pag.experiments.records import RecordStore
from pag.experiments.report import summarize_method, summarize_pair


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _comparison(
    candidate: list[dict[str, object]],
    baseline: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    result = summarize_pair(
        candidate,
        baseline,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    baseline_mean = float(np.mean([float(row["total_nfe"]) for row in baseline]))
    result["nfe_reduction"] = -float(result["nfe_difference"]["estimate"]) / baseline_mean
    return result


def _dataset_summary(
    methods: dict[str, list[dict[str, object]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    return {
        "methods": {name: summarize_method(rows) for name, rows in methods.items()},
        "residual_pag_vs_adablock": _comparison(
            methods["residual_pag"],
            methods["adablock"],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "residual_pag_vs_size_lookup": _comparison(
            methods["residual_pag"],
            methods["size_lookup"],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }


def _write_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Dataset & AdaBlock NFE & Residual PAG NFE & Reduction \\\\",
        "\\midrule",
    ]
    for model in ("llada", "dream"):
        for dataset in ("gsm8k", "math500"):
            values = summary[model][dataset]
            baseline = values["methods"]["adablock"]["mean_nfe"]
            candidate = values["methods"]["residual_pag"]["mean_nfe"]
            reduction = values["residual_pag_vs_adablock"]["nfe_reduction"]
            lines.append(
                f"{model} & {dataset} & {baseline:.2f} & {candidate:.2f} & "
                f"{100 * reduction:.1f}\\% \\\\"
            )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nfe_figure(path: Path, summary: dict[str, Any]) -> None:
    labels: list[str] = []
    values: list[float] = []
    for model in ("llada", "dream"):
        for dataset in ("gsm8k", "math500"):
            labels.append(f"{model}\n{dataset}")
            values.append(
                100 * float(summary[model][dataset]["residual_pag_vs_adablock"]["nfe_reduction"])
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 3.5))
    axis.bar(labels, values)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("NFE reduction vs AdaBlock (%)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _write_failures(
    path: Path,
    records: dict[str, dict[str, dict[str, list[dict[str, object]]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("model", "dataset", "sample_id", "category"),
        )
        writer.writeheader()
        for model, datasets in records.items():
            for dataset, methods in datasets.items():
                baseline = {row["sample_id"]: row for row in methods["adablock"]}
                for row in methods["residual_pag"]:
                    left = bool(row["grade"]["is_correct"])
                    right = bool(baseline[row["sample_id"]]["grade"]["is_correct"])
                    if left and right:
                        continue
                    category = "regression" if right else "shared_error" if not left else "recovery"
                    writer.writerow(
                        {
                            "model": model,
                            "dataset": dataset,
                            "sample_id": row["sample_id"],
                            "category": category,
                        }
                    )


def write_cross_model_report(
    run_dir: str | Path,
    *,
    identity: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
    thresholds: dict[str, float],
) -> dict[str, object]:
    root = Path(run_dir)
    output = root / "report"
    store = RecordStore(root, identity)
    all_records: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}
    summary: dict[str, Any] = {}
    for model in ("llada", "dream"):
        model_records: dict[str, dict[str, list[dict[str, object]]]] = {}
        model_summary: dict[str, Any] = {}
        for dataset, stage_name in (("gsm8k", "test_gsm8k"), ("math500", "test_math500")):
            stage = f"{stage_name}/{model}"
            methods = {
                method: store.records(stage, method)
                for method in ("adablock", "size_lookup", "residual_pag")
            }
            if not all(methods.values()):
                raise ValueError(f"incomplete cross-model report stage: {stage}")
            store.paired_records(stage, methods)
            model_records[dataset] = methods
            model_summary[dataset] = _dataset_summary(
                methods,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        aggregate = {
            method: [
                *model_records["gsm8k"][method],
                *model_records["math500"][method],
            ]
            for method in ("adablock", "size_lookup", "residual_pag")
        }
        model_summary["aggregate"] = _dataset_summary(
            aggregate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        all_records[model] = model_records
        summary[model] = model_summary
    claim_thresholds = ClaimThresholds(**thresholds)
    audit = audit_claims(summary, thresholds=claim_thresholds)
    _write_json(output / "summary.json", summary)
    _write_json(output / "claim_audit.json", audit)
    _write_table(output / "tables" / "cross_model.tex", summary)
    _write_nfe_figure(output / "figures" / "nfe_reduction.pdf", summary)
    _write_failures(output / "failure_taxonomy.csv", all_records)
    headline = (
        "Residual PAG passed every predeclared cross-model claim gate."
        if audit["headline_eligible"]
        else "Residual PAG did not pass every predeclared cross-model claim gate."
    )
    (output / "headline.tex").write_text(headline + "\n", encoding="utf-8")
    return audit
