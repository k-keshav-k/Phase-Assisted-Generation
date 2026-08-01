from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
while str(LLADA_DIR) in sys.path:
    sys.path.remove(str(LLADA_DIR))
sys.path.insert(0, str(LLADA_DIR))

if not hasattr(sys.modules.get("transformers"), "AutoTokenizer"):
    for module_name in tuple(sys.modules):
        if module_name == "transformers" or module_name.startswith("transformers."):
            del sys.modules[module_name]
for module_name in tuple(sys.modules):
    if module_name == "model" or module_name.startswith("model."):
        del sys.modules[module_name]
generate_adablock = importlib.import_module("generate_adablock")
for module_name in tuple(sys.modules):
    if module_name == "model" or module_name.startswith("model."):
        del sys.modules[module_name]


def _logits(length: int, predictions: dict[int, int]) -> torch.Tensor:
    values = torch.zeros((1, length, 10), dtype=torch.float32)
    for position, token in predictions.items():
        values[0, position, token] = 8.0
    return values


class FakeModel:
    def __init__(self, plan: list[torch.Tensor]) -> None:
        self.device = torch.device("cpu")
        self.plan = plan
        self.calls = 0

    def __call__(self, *args, **kwargs):
        del args, kwargs
        logits = self.plan[self.calls]
        self.calls += 1
        return SimpleNamespace(logits=logits, past_key_values=((torch.zeros(1),),))


class NeverStopPolicy:
    def reset_prompt(self) -> None:
        self.history = []
        self.observations = []

    def start_block(self) -> None:
        pass

    def observe(self, observation):
        self.observations.append(observation)
        return SimpleNamespace(
            should_stop=False,
            risk_score=1.0,
            safe_streak=0,
            reason="continue",
        )

    def record_realized(self, block) -> None:
        self.history.append(block)


def test_instrumented_llada_adablock_is_exact_when_policy_never_stops() -> None:
    plan = [
        _logits(5, {1: 2}),
        _logits(2, {1: 3}),
        _logits(5, {3: 4}),
        _logits(2, {1: 5}),
    ]
    kwargs = {
        "steps": 8,
        "gen_length": 4,
        "init_block_length": 2,
        "temperature": 0.0,
        "mask_id": 9,
        "threshold": 1.0,
        "delimiter_ids": [8],
        "delimiter_threshold": float("inf"),
    }
    prompt = torch.tensor([[1]], dtype=torch.long)
    baseline_model = FakeModel(plan)
    baseline_tokens, baseline_nfe, baseline_blocks = generate_adablock.generate_adablock_dual_cache(
        baseline_model,
        prompt,
        **kwargs,
    )

    policy = NeverStopPolicy()
    instrumented_model = FakeModel(plan)
    instrumented_tokens, instrumented_nfe, instrumented_blocks, schedules = (
        generate_adablock.generate_adablock_dual_cache(
            instrumented_model,
            prompt,
            **kwargs,
            risk_policy=policy,
            digit_ids_tensor=torch.tensor([2, 4]),
            delimiter_ids_tensor=torch.tensor([3, 5]),
            return_schedule_history=True,
        )
    )

    assert torch.equal(instrumented_tokens, baseline_tokens)
    assert instrumented_nfe == baseline_nfe == [2, 2]
    assert instrumented_blocks == baseline_blocks == [2, 2]
    assert instrumented_model.calls == baseline_model.calls == 4
    assert [row["actual_nfe_used"] for row in schedules] == [2, 2]
    assert len(policy.observations) == 4
