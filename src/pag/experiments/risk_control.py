from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from scipy.stats import beta, binomtest


@dataclass(frozen=True, slots=True)
class CandidateRisk:
    name: str
    losses: tuple[int, ...]
    mean_nfe: float
    protocol_identity: str = "default"


@dataclass(frozen=True, slots=True)
class CandidateAudit:
    name: str
    failures: int
    count: int
    empirical_risk: float
    upper_risk_bound: float
    pvalue: float
    corrected_cutoff: float
    certified: bool
    mean_nfe: float


@dataclass(frozen=True, slots=True)
class RiskCertificate:
    schema_version: int
    alpha: float
    familywise_delta: float
    correction: str
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


def certify_candidates(
    candidates: Sequence[CandidateRisk],
    *,
    alpha: float,
    delta: float,
) -> RiskCertificate:
    risk_level = _validate_probability(alpha, name="alpha")
    familywise_delta = _validate_probability(delta, name="delta")
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

    ordered = tuple(sorted(candidates, key=lambda item: item.name))
    corrected_cutoff = familywise_delta / len(ordered)
    audits: list[CandidateAudit] = []
    for candidate in ordered:
        failures = int(sum(candidate.losses))
        pvalue = binomial_null_pvalue(candidate.losses, alpha=risk_level)
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
                pvalue=pvalue,
                corrected_cutoff=corrected_cutoff,
                certified=pvalue <= corrected_cutoff,
                mean_nfe=float(candidate.mean_nfe),
            )
        )
    certified = [candidate for candidate in audits if candidate.certified]
    selected = (
        min(certified, key=lambda item: (item.mean_nfe, item.name)).name
        if certified
        else "full_budget"
    )
    return RiskCertificate(
        schema_version=1,
        alpha=risk_level,
        familywise_delta=familywise_delta,
        correction="bonferroni",
        protocol_identity=next(iter(identities)),
        selected=selected,
        fallback=not certified,
        candidates=tuple(audits),
    )
