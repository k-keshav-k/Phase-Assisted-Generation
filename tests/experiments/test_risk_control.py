from __future__ import annotations

import numpy as np
import pytest

from pag.experiments.risk_control import (
    CandidateRisk,
    binomial_null_pvalue,
    bounded_mean_lower_bound,
    bounded_mean_upper_tail_pvalue,
    certify_candidates,
    one_sided_upper_bound,
)


def test_invalid_candidate_is_not_certified() -> None:
    candidate = CandidateRisk("fast", losses=(1,) * 20 + (0,) * 80, mean_nfe=40.0)
    result = certify_candidates((candidate,), alpha=0.05, delta=0.05)
    assert result.selected == "full_budget"
    assert not result.candidates[0].certified
    assert result.fallback


def test_lowest_compute_certified_candidate_wins() -> None:
    candidates = (
        CandidateRisk("safe", losses=(0,) * 200, mean_nfe=50.0),
        CandidateRisk("safer", losses=(0,) * 200, mean_nfe=60.0),
    )
    result = certify_candidates(candidates, alpha=0.05, delta=0.05)
    assert result.selected == "safe"
    assert result.familywise_delta == 0.05
    assert not result.fallback
    assert all(candidate.corrected_cutoff == 0.025 for candidate in result.candidates)


def test_binomial_pvalue_and_upper_bound_known_cases() -> None:
    assert binomial_null_pvalue((0,) * 100, alpha=0.05) == pytest.approx(0.95**100)
    assert one_sided_upper_bound(0, 100, error_level=0.05) < 0.03
    assert one_sided_upper_bound(100, 100, error_level=0.05) == 1.0


def test_joint_certificate_requires_harm_and_minimum_compute_reduction() -> None:
    weak_compute = CandidateRisk(
        "weak",
        losses=(0,) * 300,
        mean_nfe=96.0,
        nfe_savings=(0.04,) * 300,
    )
    useful = CandidateRisk(
        "useful",
        losses=(0,) * 300,
        mean_nfe=80.0,
        nfe_savings=(0.20,) * 300,
    )

    result = certify_candidates(
        (weak_compute, useful),
        alpha=0.02,
        delta=0.05,
        minimum_nfe_reduction=0.05,
    )
    rows = {row.name: row for row in result.candidates}

    assert rows["weak"].harm_certified
    assert not rows["weak"].compute_certified
    assert not rows["weak"].certified
    assert rows["useful"].certified
    assert rows["useful"].lower_nfe_reduction_bound > 0.05
    assert result.selected == "useful"
    assert result.hypotheses == 4
    assert result.minimum_nfe_reduction == 0.05
    assert rows["useful"].corrected_cutoff == pytest.approx(0.05 / 4)


def test_bounded_compute_test_is_monotone_and_rejects_invalid_savings() -> None:
    strong = bounded_mean_upper_tail_pvalue((0.20,) * 300, null_mean=0.05)
    weak = bounded_mean_upper_tail_pvalue((0.06,) * 300, null_mean=0.05)
    assert strong < weak
    assert bounded_mean_lower_bound((0.20,) * 300, error_level=0.05) > 0.05
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bounded_mean_upper_tail_pvalue((-0.01, 0.10), null_mean=0.05)


def test_rejects_duplicate_candidates_and_nonbinary_losses() -> None:
    duplicate = CandidateRisk("same", losses=(0,) * 20, mean_nfe=2.0)
    with pytest.raises(ValueError, match="unique"):
        certify_candidates((duplicate, duplicate), alpha=0.05, delta=0.05)
    bad = CandidateRisk("bad", losses=(0, 2), mean_nfe=2.0)
    with pytest.raises(ValueError, match="binary"):
        certify_candidates((bad,), alpha=0.05, delta=0.05)


def test_bonferroni_selection_controls_invalid_family_in_seeded_simulation() -> None:
    rng = np.random.default_rng(20260729)
    selected_invalid = 0
    repetitions = 1000
    for _ in range(repetitions):
        candidates = tuple(
            CandidateRisk(
                f"invalid-{index}",
                losses=tuple(int(value) for value in rng.binomial(1, 0.08, size=200)),
                mean_nfe=40.0 + index,
            )
            for index in range(3)
        )
        certificate = certify_candidates(candidates, alpha=0.05, delta=0.05)
        selected_invalid += not certificate.fallback
    assert selected_invalid / repetitions <= 0.07


def test_bounded_compute_pvalue_controls_boundary_null_in_seeded_simulation() -> None:
    rng = np.random.default_rng(20260801)
    rejected = 0
    repetitions = 1000
    for _ in range(repetitions):
        savings = tuple(float(value) for value in rng.binomial(1, 0.05, size=300))
        rejected += bounded_mean_upper_tail_pvalue(savings, null_mean=0.05) <= 0.05
    assert rejected / repetitions <= 0.07
