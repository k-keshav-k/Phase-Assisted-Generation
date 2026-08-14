from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
DREAM_DIR = REPO_ROOT / "AdaBlock-dLLM" / "dream"
if str(DREAM_DIR) not in sys.path:
    sys.path.insert(0, str(DREAM_DIR))


def _install_transformers_stubs() -> None:
    transformers = ModuleType("transformers")
    transformers.__version__ = "0.0"

    generation = ModuleType("transformers.generation")
    generation_configuration = ModuleType("transformers.generation.configuration_utils")

    class GenerationConfig:
        pass

    generation_configuration.GenerationConfig = GenerationConfig

    utils = ModuleType("transformers.utils")

    class ModelOutput:
        pass

    utils.ModelOutput = ModelOutput
    utils.is_torchdynamo_compiling = lambda: False
    utils.logging = SimpleNamespace(get_logger=logging.getLogger)

    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("transformers.generation", generation)
    sys.modules.setdefault(
        "transformers.generation.configuration_utils",
        generation_configuration,
    )
    sys.modules.setdefault("transformers.utils", utils)


_install_transformers_stubs()
model_pkg = ModuleType("model")
model_pkg.__path__ = [str(DREAM_DIR / "model")]
sys.modules.setdefault("model", model_pkg)

DreamGenerationConfig = importlib.import_module(
    "model.generation_utils_adablock",
).DreamGenerationConfig
AdaBlockDreamGenerationMixin = importlib.import_module(
    "model.generation_utils_adablock",
).DreamGenerationMixin
dream_pag_module = importlib.import_module("model.generation_utils_pag")
DreamGenerationMixin = dream_pag_module.DreamGenerationMixin
PhaseTuple = importlib.import_module("phase_predict.schema").PhaseTuple


class FakeScheduler:
    def __init__(self, schedules: list[SimpleNamespace]) -> None:
        self.schedules = schedules
        self.reset_calls = 0
        self.recorded: list[tuple[int, int, float, float, float, float]] = []
        self._index = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self._index = 0
        self.recorded.clear()

    def next_schedule(self, **kwargs) -> SimpleNamespace:
        del kwargs
        schedule = self.schedules[self._index]
        self._index += 1
        return schedule

    def record_realized(
        self,
        block_size: int,
        actual_nfe_used: int,
        mean_confidence: float = 1.0,
        min_confidence: float = 1.0,
        digit_fraction: float = 0.0,
        delimiter_fraction: float = 0.0,
    ) -> None:
        self.recorded.append(
            (
                block_size,
                actual_nfe_used,
                mean_confidence,
                min_confidence,
                digit_fraction,
                delimiter_fraction,
            )
        )


class FakeModel:
    def __init__(self, logits_plan: list[torch.Tensor], scheduler: FakeScheduler) -> None:
        self.device = torch.device("cpu")
        self.logits_plan = logits_plan
        self.pag_scheduler = scheduler
        self.pag_digit_ids = torch.tensor([5, 7], dtype=torch.long)
        self.pag_delimiter_ids = torch.tensor([3, 6], dtype=torch.long)
        self.call_index = 0

    def __call__(self, *args, **kwargs):
        del args, kwargs
        logits = self.logits_plan[self.call_index]
        self.call_index += 1
        return SimpleNamespace(logits=logits, past_key_values=[(torch.zeros(1), torch.zeros(1))])

    def _compute_block_length(self, *args, **kwargs) -> int:
        del args, kwargs
        return 2


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


class NeverStopPolicy(FakeRiskPolicy):
    def observe(self, observation):
        self.observations.append(observation)
        return SimpleNamespace(
            should_stop=False,
            risk_score=1.0,
            safe_streak=0,
            reason="continue",
        )


class StableDreamModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.pag_digit_ids = torch.tensor([2, 4], dtype=torch.long)
        self.pag_delimiter_ids = torch.tensor([3, 5], dtype=torch.long)
        self.call_index = 0

    def __call__(self, tokens, *args, **kwargs):
        del args, kwargs
        self.call_index += 1
        batch, length = tokens.shape
        logits = torch.zeros((batch, length, 10), dtype=torch.float32)
        token_offset = 2 if length == 5 else 3
        for position in range(length):
            logits[:, position, position + token_offset] = 10.0 - position
        cache = [(torch.zeros((batch, 5, 1)), torch.zeros((batch, 5, 1)))]
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def _compute_block_length(self, *args, **kwargs) -> int:
        del args, kwargs
        return 4


class BatchSensitiveDreamModel(StableDreamModel):
    def __call__(self, tokens, *args, **kwargs):
        result = super().__call__(tokens, *args, **kwargs)
        if tokens.shape[0] > 1:
            result.logits.zero_()
        return result


class AlwaysDeepScorer:
    def predict_risk(self, features) -> float:
        del features
        return 0.0


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


def test_instrumented_dream_adablock_is_exact_when_policy_never_stops() -> None:
    logits_plan = [
        _make_logits(6, 8, {1: (3, 8.0)}),
        _make_logits(2, 8, {0: (4, 8.0)}),
        _make_logits(6, 8, {3: (5, 8.0)}),
        _make_logits(2, 8, {0: (6, 8.0)}),
    ]
    config = DreamGenerationConfig(
        max_length=6,
        steps=4,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    baseline = FakeModel([value.clone() for value in logits_plan], FakeScheduler([]))
    baseline._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        baseline, FakeModel
    )
    expected = baseline._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=0.8,
        block_length=2,
        dual_cache=True,
    )

    instrumented = FakeModel([value.clone() for value in logits_plan], FakeScheduler([]))
    instrumented.pag_risk_policy = NeverStopPolicy()
    instrumented._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        instrumented, FakeModel
    )
    actual = instrumented._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=0.8,
        block_length=2,
        dual_cache=True,
    )

    assert actual.sequences.tolist() == expected.sequences.tolist()
    assert actual.nfe_history == expected.nfe_history == [2, 2]
    assert actual.block_history == expected.block_history == [2, 2]
    assert len(actual.schedule_history) == 2


def test_dream_verified_speculation_matches_adablock_with_fewer_physical_nfes() -> None:
    config = DreamGenerationConfig(
        max_length=5,
        steps=8,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )
    input_ids = torch.tensor([[1]], dtype=torch.long)
    baseline = StableDreamModel()
    baseline._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        baseline,
        StableDreamModel,
    )
    expected = baseline._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=1.0,
        block_length=4,
        dual_cache=True,
    )

    speculative = StableDreamModel()
    speculative.pag_speculation_policy = RiskAdaptiveSpeculationPolicy(
        AlwaysDeepScorer(),
        max_depth=3,
        medium_depth=1,
        deep_risk_threshold=0.1,
        medium_risk_threshold=0.3,
        draft_width_multiplier=1.0,
    )
    speculative._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        speculative,
        StableDreamModel,
    )
    actual = speculative._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=1.0,
        block_length=4,
        dual_cache=True,
    )

    assert actual.sequences.tolist() == expected.sequences.tolist()
    assert expected.nfe_history == [4]
    assert actual.nfe_history == [2]
    assert speculative.call_index == 2
    assert actual.schedule_history[0]["verified_sequence_safe"] is True


def test_dream_ec_pag_audit_uses_canonical_root_and_records_work() -> None:
    config = DreamGenerationConfig(
        max_length=5,
        steps=8,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )
    input_ids = torch.tensor([[1]], dtype=torch.long)
    baseline = StableDreamModel()
    baseline._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        baseline,
        StableDreamModel,
    )
    expected = baseline._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=1.0,
        block_length=4,
        dual_cache=True,
    )
    audited = StableDreamModel()
    audited.pag_speculation_policy = EquivalenceCostPolicy.audit(depth=1, threshold=1.0)
    audited._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        audited,
        StableDreamModel,
    )
    actual = audited._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=1.0,
        block_length=4,
        dual_cache=True,
    )

    assert actual.sequences.tolist() == expected.sequences.tolist()
    assert expected.nfe_history == [4]
    assert actual.nfe_history == [7]
    steps = actual.schedule_history[0]["speculation_steps"]
    assert len(steps) == 3
    assert all(step["reference_checked"] for step in steps)
    assert all(step["evaluated_rows"] == 3 for step in steps)
    assert len(actual.schedule_history[0]["state_digests"]) == 4


def test_dream_ec_pag_unsafe_root_falls_back_to_canonical_adablock() -> None:
    config = DreamGenerationConfig(
        max_length=5,
        steps=8,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )
    input_ids = torch.tensor([[1]], dtype=torch.long)
    baseline = StableDreamModel()
    baseline._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        baseline,
        StableDreamModel,
    )
    expected = baseline._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=1.0,
        block_length=4,
        dual_cache=True,
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
    candidate = BatchSensitiveDreamModel()
    candidate.pag_speculation_policy = policy
    candidate._sample = AdaBlockDreamGenerationMixin._sample_adablock_cache.__get__(
        candidate,
        BatchSensitiveDreamModel,
    )
    actual = candidate._sample(
        input_ids,
        attention_mask=None,
        generation_config=config,
        threshold=1.0,
        block_length=4,
        dual_cache=True,
    )

    assert actual.sequences.tolist() == expected.sequences.tolist()
    steps = actual.schedule_history[0]["speculation_steps"]
    assert steps
    assert all(not step["guard_passed"] for step in steps)
    assert all(step["reference_checked"] for step in steps)
    assert all(step["canonical_fallback_rows"] == 1 for step in steps)


def test_pag_decode_uses_refinement_budget_and_force_commits_final_pass() -> None:
    schedules = [
        _make_schedule(2, 2),
        _make_schedule(2, 1),
    ]
    scheduler = FakeScheduler(schedules)
    logits_plan = [
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={
                1: (3, 8.0),
                2: (4, 0.1),
            },
        ),
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={
                2: (5, 0.1),
            },
        ),
        _make_logits(
            seq_len=6,
            vocab_size=8,
            predictions={
                3: (6, 0.1),
                4: (7, 0.1),
            },
        ),
    ]
    model = FakeModel(logits_plan, scheduler)
    model._sample = DreamGenerationMixin._sample_pag.__get__(model, FakeModel)

    generation_config = DreamGenerationConfig(
        max_length=6,
        steps=4,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    result = model._sample(
        input_ids,
        attention_mask=None,
        generation_config=generation_config,
        threshold=0.8,
        max_block_length=2,
        max_refinement_steps=4,
    )

    assert result.sequences.tolist() == [[1, 2, 3, 5, 6, 7]]
    assert result.nfe_history == [2, 1]
    assert result.block_history == [2, 2]
    assert [(row[0], row[1]) for row in scheduler.recorded] == [(2, 2), (2, 1)]
    for row in scheduler.recorded:
        assert all(0.0 <= value <= 1.0 for value in row[2:])
    assert scheduler.recorded[0][4] == 0.5
    assert scheduler.recorded[1][5] == 0.5
    assert result.schedule_history == [
        {
            "block_index": 0,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 2},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 2,
            "actual_nfe_used": 2,
            "mean_top1_confidence": scheduler.recorded[0][2],
            "min_top1_confidence": scheduler.recorded[0][3],
            "digit_fraction": 0.5,
            "delimiter_fraction": 0.5,
            "block_start": 2,
            "block_end": 4,
        },
        {
            "block_index": 1,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 1},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 1,
            "actual_nfe_used": 1,
            "mean_top1_confidence": scheduler.recorded[1][2],
            "min_top1_confidence": scheduler.recorded[1][3],
            "digit_fraction": 0.5,
            "delimiter_fraction": 0.5,
            "block_start": 4,
            "block_end": 6,
        },
    ]


def test_cached_pag_uses_adablock_boundary_and_counts_initial_forward() -> None:
    scheduler = FakeScheduler([_make_schedule(2, 1), _make_schedule(2, 1)])
    logits_plan = [
        _make_logits(6, 8, {1: (3, 8.0), 2: (5, 8.0)}),
        _make_logits(6, 8, {3: (6, 8.0), 4: (7, 8.0)}),
    ]
    model = FakeModel(logits_plan, scheduler)
    model._sample = DreamGenerationMixin._sample_pag_cache.__get__(model, FakeModel)
    generation_config = DreamGenerationConfig(
        max_length=6,
        steps=4,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )
    result = model._sample(
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=None,
        generation_config=generation_config,
        threshold=0.8,
        dual_cache=True,
        max_block_length=2,
        max_refinement_steps=4,
    )
    assert result.sequences.tolist() == [[1, 2, 3, 5, 6, 7]]
    assert result.nfe_history == [1, 1]
    assert result.block_history == [2, 2]


def test_dream_compact_observation_adapter_fields() -> None:
    logits = _make_logits(2, 8, {0: (5, 4.0), 1: (6, 3.0)})
    observation = dream_pag_module._rc_pag_observation(
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


def test_dream_policy_stop_and_shadow_label_are_recorded() -> None:
    scheduler = FakeScheduler([_make_schedule(2, 4)])
    model = FakeModel([_make_logits(4, 8, {1: (5, 4.0), 2: (6, 3.0)})], scheduler)
    model._sample = DreamGenerationMixin._sample_pag.__get__(model, FakeModel)
    model.pag_risk_policy = FakeRiskPolicy()
    model.pag_shadow_callback = lambda request: request.proposed_tokens.clone()
    generation_config = DreamGenerationConfig(
        max_length=4,
        steps=4,
        alg="confidence_threshold",
        temperature=0.0,
        return_dict_in_generate=True,
        output_history=False,
        mask_token_id=0,
    )

    result = model._sample(
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=None,
        generation_config=generation_config,
        threshold=0.99,
        max_block_length=2,
        max_refinement_steps=4,
    )

    assert result.sequences.tolist() == [[1, 2, 5, 6]]
    assert result.nfe_history == [1]
    assert len(model.pag_risk_policy.observations) == 1
    assert len(model.pag_risk_policy.realized) == 1
    assert result.schedule_history[0]["shadow_losses"] == [0]
