from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class GradeResult:
    is_correct: bool
    extracted_answer: str | None
    gold_answer: str
    error: str | None = None


_FINAL_ANSWER = re.compile(r"final\s+answer\s*:\s*([^\n\r]+)", re.IGNORECASE)


def _clean_numeric(value: str) -> str:
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "").strip()
    cleaned = cleaned.rstrip(". ")
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        cleaned = cleaned[7:-1].strip()
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)", cleaned)
    if not match:
        raise ValueError(f"not a strict numeric answer: {value!r}")
    return match.group(1)


def _fraction(value: str) -> Fraction:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(numerator) / Fraction(denominator)
    return Fraction(value)


def grade_gsm8k(text: str, gold_answer: str) -> GradeResult:
    matches = list(_FINAL_ANSWER.finditer(text))
    if not matches:
        return GradeResult(False, None, gold_answer, "missing Final answer marker")
    extracted = matches[-1].group(1).strip()
    try:
        correct = _fraction(_clean_numeric(extracted)) == _fraction(_clean_numeric(gold_answer))
    except (ValueError, ZeroDivisionError) as exc:
        return GradeResult(False, extracted, gold_answer, str(exc))
    return GradeResult(correct, extracted, gold_answer)


def grade_math500(text: str, gold_answer: str) -> GradeResult:
    try:
        from math_verify import parse, verify

        parsed_gold = parse(gold_answer, raise_on_error=True)
        parsed_prediction = parse(text, raise_on_error=True)
        if not parsed_gold or not parsed_prediction:
            return GradeResult(False, None, gold_answer, "Math-Verify extracted no answer")
        return GradeResult(
            bool(verify(parsed_gold, parsed_prediction, strict=True, raise_on_error=True)),
            str(parsed_prediction),
            gold_answer,
        )
    except Exception as exc:  # Math-Verify exposes parser-specific exception types.
        return GradeResult(False, None, gold_answer, f"{type(exc).__name__}: {exc}")
