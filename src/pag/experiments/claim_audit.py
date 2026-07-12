from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ClaimThresholds:
    minimum_nfe_reduction: float
    minimum_lookup_reduction: float
    minimum_accuracy_ci: float


def audit_claims(
    summary: dict[str, Any],
    *,
    thresholds: ClaimThresholds,
) -> dict[str, object]:
    gates: list[dict[str, object]] = []

    def add(name: str, value: float, threshold: float, *, operator: str = ">=") -> None:
        passed = value >= threshold if operator == ">=" else value > threshold
        gates.append(
            {
                "name": name,
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "passed": passed,
            }
        )

    for model in ("llada", "dream"):
        model_summary = summary[model]
        aggregate = model_summary["aggregate"]
        versus_adablock = aggregate["residual_pag_vs_adablock"]
        versus_lookup = aggregate["residual_pag_vs_size_lookup"]
        add(
            f"{model}/aggregate/adablock_nfe",
            float(versus_adablock["nfe_reduction"]),
            thresholds.minimum_nfe_reduction,
        )
        add(
            f"{model}/aggregate/adablock_accuracy",
            float(versus_adablock["accuracy_difference"]["lower"]),
            thresholds.minimum_accuracy_ci,
        )
        add(
            f"{model}/aggregate/lookup_nfe",
            float(versus_lookup["nfe_reduction"]),
            thresholds.minimum_lookup_reduction,
        )
        add(
            f"{model}/aggregate/lookup_accuracy",
            float(versus_lookup["accuracy_difference"]["lower"]),
            thresholds.minimum_accuracy_ci,
        )
        for dataset in ("gsm8k", "math500"):
            comparison = model_summary[dataset]["residual_pag_vs_adablock"]
            add(
                f"{model}/{dataset}/nfe_direction",
                float(comparison["nfe_reduction"]),
                0.0,
                operator=">",
            )
    failed = [str(gate["name"]) for gate in gates if not gate["passed"]]
    return {
        "headline_eligible": not failed,
        "gates": gates,
        "failed_gates": failed,
    }
