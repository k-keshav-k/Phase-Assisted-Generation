from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from phase_predict.schema import PhaseTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

PAGTupleScheduler = importlib.import_module("pag_predictor").PAGTupleScheduler


class FakePredictor:
    def __init__(self, outputs: list[PhaseTuple], window_size: int = 4) -> None:
        self._outputs = outputs
        self.calls: list[list[PhaseTuple]] = []
        self.config = SimpleNamespace(window_size=window_size)

    def predict(self, context):
        self.calls.append(list(context))
        return SimpleNamespace(predicted_tuple=self._outputs[len(self.calls) - 1])


def test_first_block_uses_explicit_seed_tuple() -> None:
    predictor = FakePredictor([PhaseTuple(9, 7)])
    scheduler = PAGTupleScheduler(
        predictor=predictor,
        seed_block_length=4,
        seed_refinement_steps=3,
    )

    scheduled = scheduler.next_schedule(
        remaining_tokens=32,
        max_block_length=16,
        max_refinement_steps=12,
    )

    assert scheduled.predicted_tuple == PhaseTuple(4, 3)
    assert scheduled.applied_block_size == 4
    assert scheduled.budgeted_refinement_steps == 3
    assert predictor.calls == []


def test_later_blocks_use_left_padded_realized_history() -> None:
    predictor = FakePredictor([PhaseTuple(6, 5)])
    scheduler = PAGTupleScheduler(
        predictor=predictor,
        seed_block_length=8,
        seed_refinement_steps=2,
    )

    first = scheduler.next_schedule(
        remaining_tokens=32,
        max_block_length=16,
        max_refinement_steps=12,
    )
    scheduler.record_realized(first.applied_block_size, 4)

    second = scheduler.next_schedule(
        remaining_tokens=28,
        max_block_length=16,
        max_refinement_steps=12,
    )

    assert predictor.calls == [
        [
            PhaseTuple(8, 2),
            PhaseTuple(8, 2),
            PhaseTuple(8, 2),
            PhaseTuple(8, 4),
        ]
    ]
    assert second.predicted_tuple == PhaseTuple(6, 5)


def test_scheduler_clamps_block_size_and_refinement_budget() -> None:
    predictor = FakePredictor([PhaseTuple(0, 0)])
    scheduler = PAGTupleScheduler(
        predictor=predictor,
        seed_block_length=5,
        seed_refinement_steps=2,
    )

    seed = scheduler.next_schedule(
        remaining_tokens=20,
        max_block_length=20,
        max_refinement_steps=20,
    )
    scheduler.record_realized(seed.applied_block_size, 3)

    scheduled = scheduler.next_schedule(
        remaining_tokens=5,
        max_block_length=4,
        max_refinement_steps=7,
    )

    assert scheduled.predicted_tuple == PhaseTuple(0, 0)
    assert scheduled.applied_block_size == 1
    assert scheduled.budgeted_refinement_steps == 1
