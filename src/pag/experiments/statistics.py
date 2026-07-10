from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, norm


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class CorrectnessMatrix:
    both_correct: int
    left_only: int
    right_only: int
    both_wrong: int


@dataclass(frozen=True, slots=True)
class McNemarResult:
    discordant: int
    pvalue: float


def wilson_interval(correct: int, total: int, confidence: float = 0.95) -> Interval:
    if not 0 <= correct <= total or total < 1:
        raise ValueError("Wilson interval requires 0 <= correct <= total and total > 0")
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    margin /= denominator
    return Interval(proportion, max(0.0, center - margin), min(1.0, center + margin))


def correctness_matrix(left: Sequence[bool], right: Sequence[bool]) -> CorrectnessMatrix:
    if len(left) != len(right):
        raise ValueError("paired correctness sequences must have equal length")
    return CorrectnessMatrix(
        both_correct=sum(a and b for a, b in zip(left, right, strict=True)),
        left_only=sum(a and not b for a, b in zip(left, right, strict=True)),
        right_only=sum(not a and b for a, b in zip(left, right, strict=True)),
        both_wrong=sum(not a and not b for a, b in zip(left, right, strict=True)),
    )


def exact_mcnemar(matrix: CorrectnessMatrix) -> McNemarResult:
    discordant = matrix.left_only + matrix.right_only
    pvalue = (
        1.0
        if discordant == 0
        else float(binomtest(min(matrix.left_only, matrix.right_only), discordant, 0.5).pvalue)
    )
    return McNemarResult(discordant, pvalue)


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    seed: int,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> Interval:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal nonempty sequences")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = np.asarray([statistic(differences[index]) for index in draws], dtype=np.float64)
    return Interval(
        float(statistic(differences)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def pair_records(
    left: Sequence[dict[str, object]], right: Sequence[dict[str, object]]
) -> list[tuple[dict[str, object], dict[str, object]]]:
    def keyed(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise ValueError(f"duplicate sample ID: {sample_id}")
            result[sample_id] = row
        return result

    left_by_id = keyed(left)
    right_by_id = keyed(right)
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired records have incomplete sample coverage")
    return [(left_by_id[key], right_by_id[key]) for key in sorted(left_by_id)]
