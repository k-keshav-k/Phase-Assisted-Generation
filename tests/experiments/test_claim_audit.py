from __future__ import annotations

from pag.experiments.claim_audit import ClaimThresholds, audit_claims


def _comparison(nfe_reduction: float, accuracy_lower: float = -0.01):
    return {
        "nfe_reduction": nfe_reduction,
        "accuracy_difference": {"estimate": 0.0, "lower": accuracy_lower, "upper": 0.01},
    }


def _passing_summary():
    return {
        model: {
            "aggregate": {
                "residual_pag_vs_adablock": _comparison(0.12),
                "residual_pag_vs_size_lookup": _comparison(0.04),
            },
            "gsm8k": {"residual_pag_vs_adablock": _comparison(0.11)},
            "math500": {"residual_pag_vs_adablock": _comparison(0.13)},
        }
        for model in ("llada", "dream")
    }


def test_headline_requires_every_model_and_dataset_gate() -> None:
    thresholds = ClaimThresholds(
        minimum_nfe_reduction=0.10,
        minimum_lookup_reduction=0.03,
        minimum_accuracy_ci=-0.02,
    )
    summary = _passing_summary()
    assert audit_claims(summary, thresholds=thresholds)["headline_eligible"] is True
    summary["dream"]["math500"]["residual_pag_vs_adablock"]["nfe_reduction"] = -0.01
    audit = audit_claims(summary, thresholds=thresholds)
    assert audit["headline_eligible"] is False
    assert "dream/math500/nfe_direction" in audit["failed_gates"]


def test_accuracy_ci_gate_uses_lower_bound() -> None:
    thresholds = ClaimThresholds(0.10, 0.03, -0.02)
    summary = _passing_summary()
    summary["llada"]["aggregate"]["residual_pag_vs_adablock"]["accuracy_difference"][
        "lower"
    ] = -0.021
    audit = audit_claims(summary, thresholds=thresholds)
    assert "llada/aggregate/adablock_accuracy" in audit["failed_gates"]
