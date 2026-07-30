from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from pag.experiments.statistics import pair_records, paired_bootstrap, wilson_interval

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MODELS = ("llada", "dream")
_DATASETS = ("gsm8k_test", "math500", "mbpp_sanitized", "humaneval")
_IN_DOMAIN = _DATASETS[:3]
_DEFAULT_METHODS = ("adablock", "best_nonlearned", "rc_pag_local", "rc_pag_history")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_correct(row: Mapping[str, Any]) -> bool:
    grade = row.get("grade")
    if isinstance(grade, Mapping):
        return bool(grade.get("is_correct"))
    if "is_correct" not in row:
        raise ValueError("confirmatory record is missing is_correct")
    return bool(row["is_correct"])


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot report an empty method")
    correctness = [_is_correct(row) for row in rows]
    nfe = np.asarray([float(row["total_nfe"]) for row in rows], dtype=np.float64)
    latency = np.asarray([float(row["elapsed_sec"]) for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "correct": sum(correctness),
        "accuracy": asdict(wilson_interval(sum(correctness), len(rows))),
        "mean_nfe": float(np.mean(nfe)),
        "median_nfe": float(np.median(nfe)),
        "mean_latency_sec": float(np.mean(latency)),
        "median_latency_sec": float(np.median(latency)),
    }


def _pair_summary(
    candidate: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    pairs = pair_records(candidate, baseline)
    candidate_nfe = [float(left["total_nfe"]) for left, _ in pairs]
    baseline_nfe = [float(right["total_nfe"]) for _, right in pairs]
    candidate_accuracy = [float(_is_correct(left)) for left, _ in pairs]
    baseline_accuracy = [float(_is_correct(right)) for _, right in pairs]
    candidate_latency = [float(left["elapsed_sec"]) for left, _ in pairs]
    baseline_latency = [float(right["elapsed_sec"]) for _, right in pairs]
    nfe = paired_bootstrap(
        candidate_nfe,
        baseline_nfe,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "count": len(pairs),
        "nfe_difference": asdict(nfe),
        "nfe_reduction": -nfe.estimate / float(np.mean(baseline_nfe)),
        "accuracy_difference": asdict(
            paired_bootstrap(
                candidate_accuracy,
                baseline_accuracy,
                samples=bootstrap_samples,
                seed=seed + 1,
            )
        ),
        "latency_difference": asdict(
            paired_bootstrap(
                candidate_latency,
                baseline_latency,
                samples=bootstrap_samples,
                seed=seed + 2,
            )
        ),
    }


def _validate_coverage(
    records: Mapping[str, Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]],
    expected_methods: Sequence[str],
) -> None:
    if set(records) != set(_MODELS):
        raise ValueError("report coverage must contain exactly LLaDA and Dream")
    for model in _MODELS:
        if set(records[model]) != set(_DATASETS):
            raise ValueError(f"report coverage is incomplete for {model}")
        for dataset in _DATASETS:
            methods = records[model][dataset]
            if set(methods) != set(expected_methods):
                raise ValueError(f"method coverage is incomplete for {model}/{dataset}")
            id_sets = [
                {str(row["sample_id"]) for row in methods[method]} for method in expected_methods
            ]
            if not id_sets[0] or any(ids != id_sets[0] for ids in id_sets[1:]):
                raise ValueError(f"paired coverage is incomplete for {model}/{dataset}")


def _risk_rows(certificate: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = {
        "name",
        "failures",
        "count",
        "empirical_risk",
        "upper_risk_bound",
        "pvalue",
        "corrected_cutoff",
        "certified",
        "mean_nfe",
    }
    values = []
    for candidate in certificate.get("candidates", ()):
        row = dict(candidate)
        if not required.issubset(row):
            raise ValueError("risk certificate candidate is incomplete")
        values.append(row)
    if not values:
        raise ValueError("risk certificate has no candidate audits")
    return values


def _selected_risk(
    risk_rows: Sequence[Mapping[str, Any]], token: str, *, model: str
) -> Mapping[str, Any] | None:
    prefix = f"{model}/"
    eligible = [
        row
        for row in risk_rows
        if str(row["name"]).startswith(prefix)
        and token in str(row["name"])
        and bool(row["certified"])
    ]
    return (
        min(eligible, key=lambda row: (float(row["mean_nfe"]), str(row["name"])))
        if eligible
        else None
    )


def _audit(
    summary: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    minimum_accuracy_lower_ci: float,
    bootstrap_samples: int,
    seed: int,
    records: Mapping[str, Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]],
    primary_method: str,
    require_history_frontier_ci: bool,
) -> dict[str, Any]:
    risk_rows = _risk_rows(certificate)
    selected_risks = {
        (model, variant): _selected_risk(risk_rows, variant, model=model)
        for model in _MODELS
        for variant in ("local", "history")
    }
    risk_ok = not bool(certificate.get("fallback", True)) and all(
        selected is not None for selected in selected_risks.values()
    )
    beat_adablock_models = {}
    for model in _MODELS:
        candidate_values = [
            float(row["total_nfe"])
            for dataset in _IN_DOMAIN
            for row in records[model][dataset][primary_method]
        ]
        baseline_values = [
            float(row["total_nfe"])
            for dataset in _IN_DOMAIN
            for row in records[model][dataset]["adablock"]
        ]
        beat_adablock_models[model] = float(np.mean(candidate_values)) < float(
            np.mean(baseline_values)
        )
    candidate_all = [
        float(row["total_nfe"])
        for model in _MODELS
        for dataset in _IN_DOMAIN
        for row in records[model][dataset][primary_method]
    ]
    heuristic_all = [
        float(row["total_nfe"])
        for model in _MODELS
        for dataset in _IN_DOMAIN
        for row in records[model][dataset]["best_nonlearned"]
    ]
    accuracy_lowers = [
        float(
            summary[model][dataset]["comparisons"][f"{primary_method}_vs_adablock"][
                "accuracy_difference"
            ]["lower"]
        )
        for model in _MODELS
        for dataset in _IN_DOMAIN
    ]
    gates = {
        "risk_certificate": risk_ok,
        "beat_adablock_both_models": all(beat_adablock_models.values()),
        "beat_best_nonlearned": float(np.mean(candidate_all)) < float(np.mean(heuristic_all)),
        "accuracy_noninferiority": min(accuracy_lowers) >= minimum_accuracy_lower_ci,
    }
    details: dict[str, Any] = {
        "primary_method": primary_method,
        "beat_adablock_models": beat_adablock_models,
        "minimum_accuracy_lower_ci": min(accuracy_lowers),
        "required_accuracy_lower_ci": minimum_accuracy_lower_ci,
        "history_frontier_required": require_history_frontier_ci,
    }
    if require_history_frontier_ci:
        if any(
            "rc_pag_local" not in records[model][dataset]
            or "rc_pag_history" not in records[model][dataset]
            for model in _MODELS
            for dataset in _IN_DOMAIN
        ):
            raise ValueError("history frontier requires paired local and history records")
        history_nfe: list[float] = []
        local_nfe: list[float] = []
        for model in _MODELS:
            for dataset in _IN_DOMAIN:
                pairs = pair_records(
                    records[model][dataset]["rc_pag_history"],
                    records[model][dataset]["rc_pag_local"],
                )
                history_nfe.extend(float(left["total_nfe"]) for left, _ in pairs)
                local_nfe.extend(float(right["total_nfe"]) for _, right in pairs)
        history_interval = paired_bootstrap(
            history_nfe,
            local_nfe,
            samples=bootstrap_samples,
            seed=seed + 37,
        )
        history_risk_not_worse = all(
            selected_risks[(model, "local")] is not None
            and selected_risks[(model, "history")] is not None
            and float(selected_risks[(model, "history")]["empirical_risk"])
            <= float(selected_risks[(model, "local")]["empirical_risk"])
            for model in _MODELS
        )
        gates["history_frontier"] = history_interval.upper < 0 and history_risk_not_worse
        details.update(
            {
                "history_minus_local_nfe": asdict(history_interval),
                "history_risk_not_worse": history_risk_not_worse,
            }
        )
    if bool(certificate.get("mock", False)):
        gates["non_mock_evidence"] = False
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "headline_eligible": not failed,
        "failed_gates": failed,
        "gates": gates,
        "details": details,
    }


def _latex_name(value: str) -> str:
    return value.replace("_", "\\_")


def _write_main_table(path: Path, summary: Mapping[str, Any], methods: Sequence[str]) -> None:
    lines = [
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Model & Dataset & Method & Accuracy & Mean NFE \\\\",
        "\\midrule",
    ]
    for model in _MODELS:
        for dataset in _DATASETS:
            for method in methods:
                values = summary[model][dataset]["methods"][method]
                lines.append(
                    f"{_latex_name(model)} & {_latex_name(dataset)} & {_latex_name(method)} & "
                    f"{100 * values['accuracy']['estimate']:.1f} & {values['mean_nfe']:.2f} \\\\"
                )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_calibration_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Candidate & Errors/$n$ & Risk & Upper & $p$ & Certified \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda item: str(item["name"])):
        lines.append(
            f"{_latex_name(str(row['name']))} & {row['failures']}/{row['count']} & "
            f"{float(row['empirical_risk']):.3f} & {float(row['upper_risk_bound']):.3f} & "
            f"{float(row['pvalue']):.4g} & {'yes' if row['certified'] else 'no'} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_table(path: Path, summary: Mapping[str, Any], methods: Sequence[str]) -> None:
    aggregate = summary["normalized_cross_model"]
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Variant & Normalized NFE & $\\Delta$ vs. AdaBlock \\\\",
        "\\midrule",
    ]
    for method in methods:
        ratio = float(aggregate[method])
        lines.append(f"{_latex_name(method)} & {ratio:.3f} & {100 * (ratio - 1):+.1f}\\% \\\\ ")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_estimator_table(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{llllrr}",
        "\\toprule",
        "Model & Features & Estimator & Split & Brier & AUROC \\\\",
        "\\midrule",
    ]
    for model, variants in sorted(manifest.get("models", {}).items()):
        for variant, payload in sorted(variants.items()):
            for kind, estimator in sorted(payload["estimators"].items()):
                validation = estimator["validation"]
                auc = validation["roc_auc"]
                auc_text = "--" if auc is None else f"{float(auc):.3f}"
                split = "holdout" if "holdout" in validation["split"] else "small-run"
                lines.append(
                    f"{_latex_name(model)} & {_latex_name(variant)} & "
                    f"{_latex_name(kind)} & {split} & {float(validation['brier']):.3f} & "
                    f"{auc_text} \\\\"
                )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(
    output: Path,
    summary: Mapping[str, Any],
    risk_rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    markers = {"llada": "o", "dream": "s"}
    for model in _MODELS:
        for method in methods:
            nfe = np.mean(
                [summary[model][dataset]["methods"][method]["mean_nfe"] for dataset in _IN_DOMAIN]
            )
            accuracy = np.mean(
                [
                    summary[model][dataset]["methods"][method]["accuracy"]["estimate"]
                    for dataset in _IN_DOMAIN
                ]
            )
            axis.scatter(nfe, accuracy, marker=markers[model], s=38)
            axis.annotate(f"{model}:{method}", (nfe, accuracy), fontsize=6)
    axis.set_xlabel("Mean NFE (lower is better)")
    axis.set_ylabel("Accuracy")
    figure.tight_layout()
    figure.savefig(output / "nfe_accuracy.pdf")
    plt.close(figure)

    names = [str(row["name"]) for row in risk_rows]
    nfe = [float(row["mean_nfe"]) for row in risk_rows]
    risks = [float(row["empirical_risk"]) for row in risk_rows]
    upper = [float(row["upper_risk_bound"]) for row in risk_rows]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.scatter(nfe, risks)
    for name, x_value, y_value in zip(names, nfe, risks, strict=True):
        axis.annotate(name, (x_value, y_value), fontsize=6)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Calibration mean NFE")
    axis.set_ylabel("Empirical strict prompt risk")
    figure.tight_layout()
    figure.savefig(output / "risk_compute.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    positions = np.arange(len(names))
    axis.errorbar(
        positions,
        risks,
        yerr=[np.zeros(len(names)), np.asarray(upper) - np.asarray(risks)],
        fmt="o",
        capsize=3,
    )
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label=r"$\alpha=0.05$")
    axis.set_xticks(positions, names, rotation=35, ha="right", fontsize=7)
    axis.set_ylabel("Risk with simultaneous upper bound")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reliability.pdf")
    plt.close(figure)


def _write_failures(
    path: Path,
    records: Mapping[str, Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]],
    methods: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("model", "dataset", "sample_id", "method", "category"),
        )
        writer.writeheader()
        for model in _MODELS:
            for dataset in _DATASETS:
                baseline = {
                    str(row["sample_id"]): _is_correct(row)
                    for row in records[model][dataset]["adablock"]
                }
                for method in methods[1:]:
                    for row in records[model][dataset][method]:
                        candidate = _is_correct(row)
                        reference = baseline[str(row["sample_id"])]
                        if candidate == reference:
                            continue
                        writer.writerow(
                            {
                                "model": model,
                                "dataset": dataset,
                                "sample_id": row["sample_id"],
                                "method": method,
                                "category": "regression" if reference else "recovery",
                            }
                        )


def write_rc_pag_report(
    run_dir: str | Path,
    *,
    records: Mapping[str, Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]],
    certificate: Mapping[str, Any],
    bootstrap_samples: int,
    seed: int,
    minimum_accuracy_lower_ci: float = -0.02,
    estimator_manifest: Mapping[str, Any] | None = None,
    methods: Sequence[str] = _DEFAULT_METHODS,
    primary_method: str = "rc_pag_history",
    require_history_frontier_ci: bool = True,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    methods = tuple(methods)
    if not methods or methods[0] != "adablock" or primary_method not in methods:
        raise ValueError("report methods must start with AdaBlock and include the primary method")
    _validate_coverage(records, methods)
    risk_rows = _risk_rows(certificate)
    summary: dict[str, Any] = {}
    for model_index, model in enumerate(_MODELS):
        summary[model] = {}
        for dataset_index, dataset in enumerate(_DATASETS):
            method_records = records[model][dataset]
            method_summary = {method: _summary(method_records[method]) for method in methods}
            comparisons = {
                f"{method}_vs_adablock": _pair_summary(
                    list(method_records[method]),
                    list(method_records["adablock"]),
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 100 * model_index + 10 * dataset_index,
                )
                for method in methods[1:]
            }
            summary[model][dataset] = {
                "methods": method_summary,
                "comparisons": comparisons,
            }
    normalized = {}
    for method in methods:
        ratios = [
            summary[model][dataset]["methods"][method]["mean_nfe"]
            / summary[model][dataset]["methods"]["adablock"]["mean_nfe"]
            for model in _MODELS
            for dataset in _IN_DOMAIN
        ]
        normalized[method] = float(np.mean(ratios))
    summary["normalized_cross_model"] = normalized
    audit = _audit(
        summary,
        certificate,
        minimum_accuracy_lower_ci=minimum_accuracy_lower_ci,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        records=records,
        primary_method=primary_method,
        require_history_frontier_ci=require_history_frontier_ci,
    )
    output = Path(run_dir) / "report"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    _write_json(output / "claim_audit.json", audit)
    _write_json(output / "risk_diagnostics.json", {"candidates": risk_rows})
    _write_main_table(output / "tables" / "main_results.tex", summary, methods)
    _write_calibration_table(output / "tables" / "calibration.tex", risk_rows)
    _write_ablation_table(output / "tables" / "ablations.tex", summary, methods)
    if estimator_manifest is not None:
        _write_estimator_table(output / "tables" / "estimator_ablation.tex", estimator_manifest)
    headline = (
        "RC-PAG satisfied every predeclared claim gate."
        if audit["headline_eligible"]
        else "RC-PAG is not eligible for the positive headline under the predeclared gates."
    )
    (output / "tables" / "headline.tex").write_text(headline + "\n", encoding="utf-8")
    (output / "headline.tex").write_text(headline + "\n", encoding="utf-8")
    _write_figures(figures, summary, risk_rows, methods)
    _write_failures(output / "failure_taxonomy.csv", records, methods)
    return audit
