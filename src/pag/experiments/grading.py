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
_LATEX_FRACTION = re.compile(
    r"\\(?:d?frac)\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
_NUMERIC_PREFIX = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))?"
)


def _clean_numeric(value: str) -> str:
    cleaned = value.strip().replace("\\(", "").replace("\\)", "").strip()
    cleaned = cleaned.strip("*`_ ")
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        cleaned = cleaned[7:-1].strip()
    latex_fraction = _LATEX_FRACTION.search(cleaned)
    if latex_fraction:
        return f"{latex_fraction.group(1)}/{latex_fraction.group(2)}"
    cleaned = cleaned.lstrip("$£€¥ ")
    match = _NUMERIC_PREFIX.match(cleaned)
    if not match:
        raise ValueError(f"not a strict numeric answer: {value!r}")
    return match.group(0).replace(",", "").replace(" ", "")


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
        from math_verify import LatexExtractionConfig, parse, verify

        parsed_gold = parse(
            f"${gold_answer}$",
            extraction_config=[LatexExtractionConfig()],
            raise_on_error=True,
        )
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
