from __future__ import annotations

from pag.experiments.datasets import materialize_gsm8k, stratified_math500


def _gsm_rows(count: int):
    return [
        {"question": f"Question {index}?", "answer": f"work #### {index}"} for index in range(count)
    ]


def test_development_and_confirmatory_ids_are_disjoint() -> None:
    splits = materialize_gsm8k(
        _gsm_rows(6200),
        _gsm_rows(1319),
        development=range(6000, 6200),
        confirmatory=range(400, 1319),
    )
    assert len(splits.development) == 200
    assert len(splits.full_test) == 1319
    assert len(splits.confirmatory) == 919
    assert set(splits.development_ids).isdisjoint(splits.full_test_ids)


def test_math_subset_is_reproducible_and_covers_strata() -> None:
    rows = [
        {
            "problem": f"Problem {index}",
            "answer": str(index),
            "subject": subject,
            "level": level,
            "unique_id": f"m-{index:03d}",
        }
        for index, (subject, level) in enumerate(
            [
                (subject, level)
                for subject in ("algebra", "geometry")
                for level in (1, 2)
                for _ in range(10)
            ]
        )
    ]
    first = stratified_math500(rows, sample_size=20, seed=20260710)
    second = stratified_math500(rows, sample_size=20, seed=20260710)
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert {(row.subject, row.level) for row in first} == {
        ("algebra", 1),
        ("algebra", 2),
        ("geometry", 1),
        ("geometry", 2),
    }
