from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

variants = importlib.import_module("scheduler_variants")


def _next(scheduler, block_size: int):
    return scheduler.next_schedule(
        block_size=block_size,
        remaining_tokens=64,
        max_block_length=64,
        max_refinement_steps=64,
    )


def test_constant_scheduler_uses_trace_median_after_seed() -> None:
    scheduler = variants.ConstantBudgetScheduler(
        seed_budget=7,
        content_budget=5,
        delimiter_budget=1,
    )
    assert _next(scheduler, 16).budgeted_refinement_steps == 7
    scheduler.record_realized(16, 6)
    assert _next(scheduler, 16).budgeted_refinement_steps == 5


def test_previous_nfe_uses_last_content_block() -> None:
    scheduler = variants.PreviousNFEScheduler(seed_budget=7)
    _next(scheduler, 16)
    scheduler.record_realized(16, 9)
    assert _next(scheduler, 1).budgeted_refinement_steps == 1
    scheduler.record_realized(1, 1, delimiter_fraction=1.0)
    assert _next(scheduler, 12).budgeted_refinement_steps == 9


def test_trace_stats_separate_delimiters_and_sizes() -> None:
    stats = variants.derive_trace_budget_stats(
        [
            [
                {"block_size": 16, "nfe": 8, "delimiter_fraction": 0},
                {"block_size": 16, "nfe": 4, "delimiter_fraction": 0},
                {"block_size": 1, "nfe": 1, "delimiter_fraction": 1},
            ]
        ]
    )
    assert stats.content_median == 6
    assert stats.delimiter_median == 1
    assert stats.by_size == {16: 6, 1: 1}
