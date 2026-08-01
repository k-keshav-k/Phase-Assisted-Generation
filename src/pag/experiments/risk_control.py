from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from scipy.stats import beta, binom, binomtest


@dataclass(frozen=True, slots=True)
class CandidateRisk:
    name: str
    losses: tuple[int, ...]
    mean_nfe: float
    protocol_identity: str = "default"
    nfe_savings: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateAudit:
    name: str
    failures: int
    count: int
    empirical_risk: float
    upper_risk_bound: float
    pvalue: float
    harm_pvalue: float
    compute_pvalue: float | None
    corrected_cutoff: float
    harm_certified: bool
    compute_certified: bool
    certified: bool
    mean_nfe: float
    empirical_nfe_reduction: float | None
    lower_nfe_reduction_bound: float | None


@dataclass(frozen=True, slots=True)
class RiskCertificate:
    schema_version: int
    alpha: float
    familywise_delta: float
    correction: str
    hypotheses: int
    minimum_nfe_reduction: float | None
    protocol_identity: str
    selected: str
    fallback: bool
    candidates: tuple[CandidateAudit, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_probability(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1)")
    return number


def binomial_null_pvalue(losses: Sequence[int], *, alpha: float) -> float:
    risk_level = _validate_probability(alpha, name="alpha")
    if not losses:
        raise ValueError("binomial risk test requires losses")
    if any(value not in (0, 1, False, True) for value in losses):
        raise ValueError("prompt losses must be binary")
    failures = int(sum(losses))
    return float(
        binomtest(
            failures,
            len(losses),
            p=risk_level,
            alternative="less",
        ).pvalue
    )


def one_sided_upper_bound(
    failures: int,
    total: int,
    *,
    error_level: float,
) -> float:
    gamma = _validate_probability(error_level, name="error_level")
    if total < 1 or not 0 <= failures <= total:
        raise ValueError("binomial upper bound requires 0 <= failures <= total and total > 0")
    if failures == total:
        return 1.0
    return float(beta.ppf(1.0 - gamma, failures + 1, total - failures))


def _validate_bounded(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("bounded mean test requires observations")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("bounded mean observations must be finite and in [0, 1]")
    return normalized


def bounded_mean_upper_tail_pvalue(
    values: Sequence[float],
    *,
    null_mean: float,
) -> float:
    """Conservative Hoeffding--Bentkus p-value for H0: E[X] <= null_mean.

    Observations must be independent and lie in [0, 1]. The integer binomial threshold is rounded
    down, which is conservative for non-integer sums.
    """

    observations = _validate_bounded(values)
    target = float(null_mean)
    if not math.isfinite(target) or not 0.0 <= target < 1.0:
        raise ValueError("null_mean must be finite and in [0, 1)")
    count = len(observations)
    empirical = float(sum(observations) / count)
    if empirical <= target:
        return 1.0
    hoeffding = math.exp(-2.0 * count * (empirical - target) ** 2)
    threshold = max(0, min(count, math.floor(sum(observations) + 1e-12)))
    bentkus = math.e * float(binom.sf(threshold - 1, count, target))
    return float(min(1.0, hoeffding, bentkus))


def bounded_mean_lower_bound(
    values: Sequence[float],
    *,
    error_level: float,
) -> float:
    """Invert the bounded-mean test to obtain a one-sided lower confidence bound."""

    observations = _validate_bounded(values)
    gamma = _validate_probability(error_level, name="error_level")
    empirical = float(sum(observations) / len(observations))
    if bounded_mean_upper_tail_pvalue(observations, null_mean=0.0) > gamma:
        return 0.0
    lower = 0.0
    upper = empirical
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        if bounded_mean_upper_tail_pvalue(observations, null_mean=midpoint) <= gamma:
            lower = midpoint
        else:
            upper = midpoint
    return float(lower)


def certify_candidates(
    candidates: Sequence[CandidateRisk],
    *,
    alpha: float,
    delta: float,
    minimum_nfe_reduction: float | None = None,
) -> RiskCertificate:
    risk_level = _validate_probability(alpha, name="alpha")
    familywise_delta = _validate_probability(delta, name="delta")
    compute_target = None if minimum_nfe_reduction is None else float(minimum_nfe_reduction)
    if compute_target is not None and (
        not math.isfinite(compute_target) or not 0.0 < compute_target < 1.0
    ):
        raise ValueError("minimum_nfe_reduction must be finite and in (0, 1)")
    if not candidates:
        raise ValueError("risk certification requires a predeclared candidate family")
    names = [candidate.name for candidate in candidates]
    if any(not name for name in names):
        raise ValueError("candidate names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    identities = {candidate.protocol_identity for candidate in candidates}
    if len(identities) != 1 or not next(iter(identities)):
        raise ValueError("candidates must share one non-empty protocol identity")
    for candidate in candidates:
        if not candidate.losses:
            raise ValueError(f"candidate {candidate.name} has no calibration losses")
        if any(value not in (0, 1, False, True) for value in candidate.losses):
            raise ValueError("prompt losses must be binary")
        if not math.isfinite(candidate.mean_nfe) or candidate.mean_nfe <= 0.0:
            raise ValueError("candidate mean_nfe must be finite and positive")
        if compute_target is not None:
            savings = _validate_bounded(candidate.nfe_savings)
            if len(savings) != len(candidate.losses):
                raise ValueError("paired harm and NFE observations must have equal counts")

    ordered = tuple(sorted(candidates, key=lambda item: item.name))
    hypotheses = len(ordered) * (2 if compute_target is not None else 1)
    corrected_cutoff = familywise_delta / hypotheses
    audits: list[CandidateAudit] = []
    for candidate in ordered:
        failures = int(sum(candidate.losses))
        harm_pvalue = binomial_null_pvalue(candidate.losses, alpha=risk_level)
        harm_certified = harm_pvalue <= corrected_cutoff
        compute_pvalue = (
            bounded_mean_upper_tail_pvalue(
                candidate.nfe_savings,
                null_mean=compute_target,
            )
            if compute_target is not None
            else None
        )
        compute_certified = compute_pvalue is None or compute_pvalue <= corrected_cutoff
        lower_reduction = (
            bounded_mean_lower_bound(
                candidate.nfe_savings,
                error_level=corrected_cutoff,
            )
            if compute_target is not None
            else None
        )
        joint_pvalue = harm_pvalue if compute_pvalue is None else max(harm_pvalue, compute_pvalue)
        audits.append(
            CandidateAudit(
                name=candidate.name,
                failures=failures,
                count=len(candidate.losses),
                empirical_risk=failures / len(candidate.losses),
                upper_risk_bound=one_sided_upper_bound(
                    failures,
                    len(candidate.losses),
                    error_level=corrected_cutoff,
                ),
                pvalue=joint_pvalue,
                harm_pvalue=harm_pvalue,
                compute_pvalue=compute_pvalue,
                corrected_cutoff=corrected_cutoff,
                harm_certified=harm_certified,
                compute_certified=compute_certified,
                certified=harm_certified and compute_certified,
                mean_nfe=float(candidate.mean_nfe),
                empirical_nfe_reduction=(
                    float(sum(candidate.nfe_savings) / len(candidate.nfe_savings))
                    if compute_target is not None
                    else None
                ),
                lower_nfe_reduction_bound=lower_reduction,
            )
        )
    certified = [candidate for candidate in audits if candidate.certified]
    selected = (
        min(certified, key=lambda item: (item.mean_nfe, item.name)).name
        if certified
        else "full_budget"
    )
    return RiskCertificate(
        schema_version=2 if compute_target is not None else 1,
        alpha=risk_level,
        familywise_delta=familywise_delta,
        correction="bonferroni",
        hypotheses=hypotheses,
        minimum_nfe_reduction=compute_target,
        protocol_identity=next(iter(identities)),
        selected=selected,
        fallback=not certified,
        candidates=tuple(audits),
    )
