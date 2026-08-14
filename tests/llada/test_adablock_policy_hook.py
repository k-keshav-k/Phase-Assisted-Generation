from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from pag.experiments.rc_pag_equivalence import (
    EquivalenceCostArtifact,
    EquivalenceCostPolicy,
    EquivalenceEnvelope,
)
from pag.experiments.rc_pag_speculation import (
    RiskAdaptiveSpeculationPolicy,
    SpeculationPlan,
)

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


class StableRankingModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls = 0

    def __call__(self, tokens, **kwargs):
        del kwargs
        self.calls += 1
        batch, length = tokens.shape
        logits = torch.zeros((batch, length, 10), dtype=torch.float32)
        full_sequence = length == 5
        for position in range(length):
            token = position + 1 if full_sequence else position + 2
            logits[:, position, token] = 10.0 - position
        cache = ((torch.zeros((batch, 1, 1, 1)), torch.zeros((batch, 1, 1, 1))),)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class BatchSensitiveRankingModel(StableRankingModel):
    def __call__(self, tokens, **kwargs):
        result = super().__call__(tokens, **kwargs)
        if tokens.shape[0] > 1:
            result.logits.zero_()
        return result


class AlwaysDeepScorer:
    def predict_risk(self, features) -> float:
        del features
        return 0.0


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


def test_verified_speculation_matches_adablock_and_reduces_physical_nfe() -> None:
    kwargs = {
        "steps": 8,
        "gen_length": 4,
        "init_block_length": 4,
        "temperature": 0.0,
        "mask_id": 9,
        "threshold": 1.0,
        "delimiter_ids": [8],
        "delimiter_threshold": float("inf"),
    }
    prompt = torch.tensor([[1]], dtype=torch.long)
    baseline_model = StableRankingModel()
    baseline_tokens, baseline_nfe, _ = generate_adablock.generate_adablock_dual_cache(
        baseline_model,
        prompt,
        **kwargs,
    )
    policy = RiskAdaptiveSpeculationPolicy(
        AlwaysDeepScorer(),
        max_depth=3,
        medium_depth=1,
        deep_risk_threshold=0.1,
        medium_risk_threshold=0.3,
        draft_width_multiplier=1.0,
    )
    speculative_model = StableRankingModel()
    speculative_tokens, speculative_nfe, _, schedules = (
        generate_adablock.generate_adablock_dual_cache(
            speculative_model,
            prompt,
            **kwargs,
            speculation_policy=policy,
            return_schedule_history=True,
        )
    )

    assert torch.equal(speculative_tokens, baseline_tokens)
    assert baseline_nfe == [4]
    assert speculative_nfe == [2]
    assert speculative_model.calls == 2
    assert schedules[0]["verified_sequence_safe"] is True
    assert schedules[0]["speculative_nfe_saved"] >= 2


def test_ec_pag_audit_uses_canonical_root_and_records_honest_work() -> None:
    kwargs = {
        "steps": 8,
        "gen_length": 4,
        "init_block_length": 4,
        "temperature": 0.0,
        "mask_id": 9,
        "threshold": 1.0,
        "delimiter_ids": [8],
        "delimiter_threshold": float("inf"),
    }
    prompt = torch.tensor([[1]], dtype=torch.long)
    baseline = StableRankingModel()
    expected, baseline_nfe, _ = generate_adablock.generate_adablock_dual_cache(
        baseline, prompt, **kwargs
    )
    audited = StableRankingModel()
    actual, audit_nfe, _, schedules = generate_adablock.generate_adablock_dual_cache(
        audited,
        prompt,
        **kwargs,
        speculation_policy=EquivalenceCostPolicy.audit(depth=1, threshold=1.0),
        return_schedule_history=True,
    )

    assert torch.equal(actual, expected)
    assert baseline_nfe == [4]
    assert audit_nfe == [7]
    steps = schedules[0]["speculation_steps"]
    assert len(steps) == 3
    assert all(step["reference_checked"] for step in steps)
    assert all(step["canonical_fallback_rows"] == 1 for step in steps)
    assert all(step["evaluated_rows"] == 3 for step in steps)
    assert len(schedules[0]["state_digests"]) == 4


def test_ec_pag_unsafe_batched_root_falls_back_to_canonical_adablock() -> None:
    kwargs = {
        "steps": 8,
        "gen_length": 4,
        "init_block_length": 4,
        "temperature": 0.0,
        "mask_id": 9,
        "threshold": 1.0,
        "delimiter_ids": [8],
        "delimiter_threshold": float("inf"),
    }
    prompt = torch.tensor([[1]], dtype=torch.long)
    expected, _, _ = generate_adablock.generate_adablock_dual_cache(
        StableRankingModel(), prompt, **kwargs
    )
    artifact = EquivalenceCostArtifact(
        schema_version=1,
        fingerprint={"gpu": "test"},
        envelopes={2: EquivalenceEnvelope(0.01, 0.001)},
        cost_rules=(),
        safety_inflation=1.25,
        minimum_bin_count=8,
        minimum_acceptance_lcb=0.8,
        artifact_hash="test",
    )
    policy = EquivalenceCostPolicy.production(artifact, threshold=1.0)
    policy.choose = lambda observation, last_transfer_count: SpeculationPlan(  # type: ignore[method-assign]
        risk_score=0.0,
        depth=1,
        draft_width=1,
        reason="test_depth_1",
    )
    actual, _, _, schedules = generate_adablock.generate_adablock_dual_cache(
        BatchSensitiveRankingModel(),
        prompt,
        **kwargs,
        speculation_policy=policy,
        return_schedule_history=True,
    )

    assert torch.equal(actual, expected)
    steps = schedules[0]["speculation_steps"]
    assert steps
    assert all(not step["guard_passed"] for step in steps)
    assert all(step["reference_checked"] for step in steps)
    assert all(step["canonical_fallback_rows"] == 1 for step in steps)
