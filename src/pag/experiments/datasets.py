from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pag.experiments.config import ExperimentConfig, inclusive_range


@dataclass(frozen=True, slots=True)
class ExperimentSample:
    sample_id: str
    dataset: str
    prompt: str
    gold_answer: str
    subject: str | None = None
    level: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GSM8KSplits:
    development: tuple[ExperimentSample, ...]
    full_test: tuple[ExperimentSample, ...]
    confirmatory: tuple[ExperimentSample, ...]

    @property
    def development_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.development)

    @property
    def full_test_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.full_test)


@dataclass(frozen=True, slots=True)
class FreshGSM8KSplits:
    calibration: tuple[ExperimentSample, ...]
    test: tuple[ExperimentSample, ...]

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.calibration)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.test)


def _gsm8k_gold(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError("GSM8K answer is missing the #### final-answer marker")
    return answer.rsplit(marker, 1)[1].strip()


def _gsm8k_sample(row: Mapping[str, Any], *, split: str, index: int) -> ExperimentSample:
    question = str(row["question"]).strip()
    prompt = (
        f"{question}\n\nSolve the problem step by step. "
        "End with a line formatted exactly as Final answer: <number>."
    )
    return ExperimentSample(
        sample_id=f"gsm8k_{split}_{index:04d}",
        dataset="gsm8k",
        prompt=prompt,
        gold_answer=_gsm8k_gold(str(row["answer"])),
        metadata={"split": split, "index": index},
    )


def materialize_gsm8k(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    development: Iterable[int],
    confirmatory: Iterable[int],
) -> GSM8KSplits:
    development_indices = tuple(development)
    confirmatory_indices = tuple(confirmatory)
    if not development_indices or max(development_indices) >= len(train_rows):
        raise ValueError("GSM8K development indices exceed the training split")
    if not confirmatory_indices or max(confirmatory_indices) >= len(test_rows):
        raise ValueError("GSM8K confirmatory indices exceed the test split")
    development_samples = tuple(
        _gsm8k_sample(train_rows[index], split="train", index=index)
        for index in development_indices
    )
    full_test = tuple(
        _gsm8k_sample(row, split="test", index=index) for index, row in enumerate(test_rows)
    )
    confirmatory_samples = tuple(full_test[index] for index in confirmatory_indices)
    if set(item.sample_id for item in development_samples) & set(
        item.sample_id for item in full_test
    ):
        raise ValueError("GSM8K development and test IDs overlap")
    return GSM8KSplits(development_samples, full_test, confirmatory_samples)


def materialize_fresh_gsm8k(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    calibration: Iterable[int],
    test: Iterable[int],
) -> FreshGSM8KSplits:
    calibration_indices = tuple(calibration)
    test_indices = tuple(test)
    all_indices = (*calibration_indices, *test_indices)
    if not calibration_indices or not test_indices:
        raise ValueError("fresh GSM8K calibration and test splits must be non-empty")
    if min(all_indices) < 0 or max(all_indices) >= len(train_rows):
        raise ValueError("fresh GSM8K indices exceed the training split")
    if set(calibration_indices) & set(test_indices):
        raise ValueError("fresh GSM8K calibration and test indices overlap")
    calibration_rows = tuple(
        _gsm8k_sample(train_rows[index], split="train", index=index)
        for index in calibration_indices
    )
    test_rows = tuple(
        _gsm8k_sample(train_rows[index], split="train", index=index) for index in test_indices
    )
    return FreshGSM8KSplits(calibration_rows, test_rows)


def _allocate_strata(
    counts: dict[tuple[str, int], int], sample_size: int
) -> dict[tuple[str, int], int]:
    total = sum(counts.values())
    if sample_size < len(counts):
        raise ValueError("sample_size must cover every MATH-500 subject/level stratum")
    if sample_size > total:
        raise ValueError("sample_size exceeds MATH-500 rows")
    allocation = {key: 1 for key in counts}
    remaining = sample_size - len(counts)
    weights = {key: remaining * count / total for key, count in counts.items()}
    for key, weight in weights.items():
        allocation[key] += min(counts[key] - 1, int(weight))
    left = sample_size - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (weights[key] - int(weights[key]), counts[key], key),
        reverse=True,
    )
    while left:
        progressed = False
        for key in order:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                left -= 1
                progressed = True
                if not left:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate the requested MATH-500 sample")
    return allocation


def stratified_math500(
    rows: Sequence[Mapping[str, Any]], *, sample_size: int, seed: int
) -> tuple[ExperimentSample, ...]:
    strata: dict[tuple[str, int], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(str(row["subject"]), int(row["level"]))].append((index, row))
    allocation = _allocate_strata({key: len(values) for key, values in strata.items()}, sample_size)
    rng = random.Random(seed)
    selected: list[ExperimentSample] = []
    for key in sorted(strata):
        candidates = list(strata[key])
        rng.shuffle(candidates)
        for index, row in sorted(candidates[: allocation[key]], key=lambda value: value[0]):
            problem = str(row["problem"]).strip()
            unique_id = str(row.get("unique_id", f"math500_{index:04d}"))
            selected.append(
                ExperimentSample(
                    sample_id=unique_id,
                    dataset="math500",
                    prompt=(
                        f"{problem}\n\nSolve the problem step by step and put the final answer "
                        "inside \\boxed{}."
                    ),
                    gold_answer=str(row["answer"]),
                    subject=key[0],
                    level=key[1],
                    metadata={"index": index},
                )
            )
    return tuple(sorted(selected, key=lambda item: item.sample_id))


def complement_math500(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_ids: set[str],
) -> tuple[ExperimentSample, ...]:
    all_samples = stratified_math500(rows, sample_size=len(rows), seed=0)
    known_ids = {sample.sample_id for sample in all_samples}
    unknown = excluded_ids - known_ids
    if unknown:
        raise ValueError(f"excluded MATH-500 IDs are unknown: {sorted(unknown)}")
    return tuple(sample for sample in all_samples if sample.sample_id not in excluded_ids)


def load_datasets(config: ExperimentConfig) -> tuple[GSM8KSplits, tuple[ExperimentSample, ...]]:
    from datasets import load_dataset

    gsm = load_dataset(
        config.gsm8k.path,
        config.gsm8k.config,
        revision=config.gsm8k.revision,
    )
    math = load_dataset(config.math500.path, revision=config.math500.revision, split="test")
    if config.gsm8k.development_indices is None or config.gsm8k.confirmatory_indices is None:
        raise ValueError("GSM8K index ranges are required")
    if config.math500.sample_size is None:
        raise ValueError("MATH-500 sample_size is required")
    gsm_splits = materialize_gsm8k(
        gsm["train"],
        gsm["test"],
        development=inclusive_range(config.gsm8k.development_indices),
        confirmatory=inclusive_range(config.gsm8k.confirmatory_indices),
    )
    return gsm_splits, stratified_math500(
        math, sample_size=config.math500.sample_size, seed=config.seed
    )
