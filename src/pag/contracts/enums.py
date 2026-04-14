from __future__ import annotations

from enum import StrEnum


class StageName(StrEnum):
    BASELINE = "baseline"
    PHASES = "phases"
    SCHEDULER = "scheduler"
    EVALUATION = "evaluation"


class PhaseLabel(StrEnum):
    EASY = "easy"
    HARD = "hard"

