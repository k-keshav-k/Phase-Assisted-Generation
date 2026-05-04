from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

generate_pag = importlib.import_module("generate_pag").generate_pag
PhaseTuple = importlib.import_module("phase_predict.schema").PhaseTuple


class FakeScheduler:
    def __init__(self, schedules: list[SimpleNamespace]) -> None:
        self.schedules = schedules
        self.reset_calls = 0
        self.recorded: list[tuple[int, int]] = []
        self._index = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.recorded.clear()
        self._index = 0

    def next_schedule(self, **kwargs) -> SimpleNamespace:
        del kwargs
        schedule = self.schedules[self._index]
        self._index += 1
        return schedule

    def record_realized(self, block_size: int, actual_nfe_used: int, **kwargs: object) -> None:
        self.recorded.append((block_size, actual_nfe_used))


class FakeModel:
    def __init__(self, logits_plan: list[torch.Tensor]) -> None:
        self.device = torch.device("cpu")
        self.logits_plan = logits_plan
        self.call_index = 0

    def __call__(self, *args, **kwargs):
        del args, kwargs
        logits = self.logits_plan[self.call_index]
        self.call_index += 1
        return SimpleNamespace(logits=logits, past_key_values=[(torch.zeros(1), torch.zeros(1))])


def _make_schedule(block_size: int, refinement_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        predicted_tuple=PhaseTuple(block_size, refinement_steps),
        applied_block_size=block_size,
        budgeted_refinement_steps=refinement_steps,
    )


def _make_logits(
    seq_len: int,
    vocab_size: int,
    predictions: dict[int, tuple[int, float]],
) -> torch.Tensor:
    logits = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32)
    for position, (token_id, score) in predictions.items():
        logits[0, position, token_id] = score
    return logits


def test_generate_pag_uses_refinement_budget_and_force_commits_final_pass() -> None:
    scheduler = FakeScheduler(
        [
            _make_schedule(2, 2),
            _make_schedule(2, 1),
        ]
    )
    logits_plan = [
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={
                2: (3, 8.0),
                3: (4, 0.1),
            },
        ),
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={
                3: (5, 0.1),
            },
        ),
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={
                4: (6, 0.1),
                5: (7, 0.1),
            },
        ),
    ]
    model = FakeModel(logits_plan)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    result, nfe_history, block_history, schedule_history = generate_pag(
        model,
        input_ids,
        scheduler,
        steps=4,
        gen_length=4,
        threshold=0.8,
        max_block_length=2,
        max_refinement_steps=4,
    )

    assert result.tolist() == [[1, 2, 3, 5, 6, 7]]
    assert nfe_history == [2, 1]
    assert block_history == [2, 2]
    assert scheduler.recorded == [(2, 2), (2, 1)]
    assert schedule_history == [
        {
            "block_index": 0,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 2},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 2,
            "actual_nfe_used": 2,
            "block_start": 2,
            "block_end": 4,
        },
        {
            "block_index": 1,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 1},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 1,
            "actual_nfe_used": 1,
            "block_start": 4,
            "block_end": 6,
        },
    ]
