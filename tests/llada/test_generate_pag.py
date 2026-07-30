from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

generate_pag_module = importlib.import_module("generate_pag")
generate_pag = generate_pag_module.generate_pag
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

    def record_realized(self, block_size: int, actual_nfe_used: int, *metrics: float) -> None:
        del metrics
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


class FakeRiskPolicy:
    def __init__(self) -> None:
        self.observations = []
        self.realized = []

    def reset_prompt(self) -> None:
        self.observations.clear()
        self.realized.clear()

    def start_block(self) -> None:
        pass

    def observe(self, observation):
        self.observations.append(observation)
        return SimpleNamespace(
            should_stop=True,
            risk_score=0.01,
            safe_streak=1,
            reason="risk_certified_candidate",
        )

    def record_realized(self, block) -> None:
        self.realized.append(block)


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
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={5: (7, 0.1)},
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
    assert nfe_history == [2, 2]
    assert block_history == [2, 2]
    assert scheduler.recorded == [(2, 2), (2, 2)]
    assert schedule_history == [
        {
            "block_index": 0,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 2},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 2,
            "actual_nfe_used": 2,
            "block_start": 2,
            "block_end": 4,
            "exit_reason": "complete",
        },
        {
            "block_index": 1,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 1},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 1,
            "actual_nfe_used": 2,
            "block_start": 4,
            "block_end": 6,
            "exit_reason": "complete",
        },
    ]


def test_hard_cap_mode_force_commits_at_budget() -> None:
    decision = importlib.import_module("generate_pag").decide_budget_enforcement(
        mode="hard_cap",
        nfe=2,
        budget=2,
        max_steps=4,
        confident=False,
        stable=False,
        complete=False,
    )
    assert decision.force_commit
    assert decision.reason == "hard_budget"


def test_invalid_enforcement_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported enforcement mode"):
        importlib.import_module("generate_pag").decide_budget_enforcement(
            mode="unknown",
            nfe=1,
            budget=2,
            max_steps=4,
            confident=False,
            stable=False,
            complete=False,
        )


def test_llada_compact_observation_adapter_fields() -> None:
    logits = _make_logits(2, 8, {0: (5, 4.0), 1: (6, 3.0)})
    observation = generate_pag_module._rc_pag_observation(
        logits,
        torch.tensor([[0, 6]]),
        mask_token_id=0,
        step_index=3,
        digit_ids=torch.tensor([5]),
        delimiter_ids=torch.tensor([6]),
    )

    assert observation.block_size == 2
    assert observation.masked == (True, False)
    assert observation.token_ids == (5, 6)
    assert observation.step_index == 3


def test_llada_policy_stop_and_shadow_label_are_recorded() -> None:
    scheduler = FakeScheduler([_make_schedule(2, 4)])
    model = FakeModel([_make_logits(4, 8, {2: (5, 4.0), 3: (6, 3.0)})])
    policy = FakeRiskPolicy()

    result, nfe_history, _, schedules = generate_pag(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        scheduler,
        steps=4,
        gen_length=2,
        threshold=0.99,
        max_block_length=2,
        max_refinement_steps=4,
        risk_policy=policy,
        shadow_callback=lambda request: request.proposed_tokens.clone(),
    )

    assert result.tolist() == [[1, 2, 5, 6]]
    assert nfe_history == [1]
    assert len(policy.observations) == 1
    assert len(policy.realized) == 1
    assert schedules[0]["exit_reason"] == "risk_policy"
    assert schedules[0]["shadow_losses"] == [0]
