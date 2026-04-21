from __future__ import annotations

import pytest

import phase_cpd.trace_jobs.run_dream_trace_dump as dream_trace_dump
from phase_cpd.trace_jobs.dream_runtime import (
    DreamGenerationConfig,
    _load_dream_model,
    _normalize_hook_step,
    _prompt_seed,
    _resolve_delimiter_features,
    _selected_token_stats,
    _slice_generated_canvas,
    _special_token_ids,
    _suppress_special_tokens_before_min_new_tokens,
    _truncate_generated_ids,
)
from phase_cpd.trace_jobs.run_dream_trace_dump import (
    _normalize_payload,
    _resolve_trace_profiles,
)


class _TorchStub:
    float32 = None

    @staticmethod
    def full_like(tensor, fill_value):
        import torch

        return torch.full_like(tensor, fill_value)


class _OutOfMemoryError(RuntimeError):
    pass


class _CudaStub:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        del seed

    @staticmethod
    def empty_cache() -> None:
        return None


class _TorchLoadStub:
    OutOfMemoryError = _OutOfMemoryError
    cuda = _CudaStub()


class _ModelStub:
    def __init__(self, *, raise_oom_on_to: bool = False) -> None:
        self.raise_oom_on_to = raise_oom_on_to
        self.to_calls: list[tuple[str, object]] = []
        self.eval_called = False

    def to(self, *, device: str, dtype):
        self.to_calls.append((device, dtype))
        if self.raise_oom_on_to:
            raise _OutOfMemoryError("oom")
        return self

    def eval(self):
        self.eval_called = True
        return self


class _AutoModelStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.first_model = _ModelStub(raise_oom_on_to=True)
        self.second_model = _ModelStub()

    def from_pretrained(self, model_name: str, **kwargs):
        self.calls.append({"model_name": model_name, **kwargs})
        if len(self.calls) == 1:
            return self.first_model
        return self.second_model


class _TokenizerStub:
    mask_token_id = 99
    mask_token = "<|mask|>"
    eos_token_id = None
    pad_token_id = None

    _DECODE_MAP = {
        11: "A",
        12: "B",
        13: ".",
        14: "\n",
        99: "<|mask|>",
    }

    def decode(
        self,
        token_ids,
        *,
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        parts = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.mask_token_id:
                continue
            parts.append(self._DECODE_MAP[int(token_id)])
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == ".":
            return [13]
        if text == "\n":
            return [14]
        if text == "..":
            return [13, 13]
        raise KeyError(text)


def test_normalize_hook_step_skips_none() -> None:
    assert _normalize_hook_step(None) is None


def test_normalize_hook_step_accepts_numeric_values() -> None:
    assert _normalize_hook_step(3) == 3
    assert _normalize_hook_step("4") == 4


def test_selected_token_stats_returns_entropy_and_top2() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    token_ids = torch.tensor([[0]], dtype=torch.long)

    selected_logits, selected_probs, top2_probs, entropies = _selected_token_stats(
        _TorchStub(),
        logits,
        token_ids,
    )

    assert selected_logits[0] == 2.0
    assert 0.0 < selected_probs[0] < 1.0
    assert 0.0 < top2_probs[0] < selected_probs[0]
    assert entropies[0] > 0.0


def test_slice_generated_canvas_drops_prompt_prefix() -> None:
    torch = pytest.importorskip("torch")
    token_canvas = torch.tensor([[1, 2, 3, 11, 12, 13]], dtype=torch.long)

    generated = _slice_generated_canvas(
        token_canvas=token_canvas,
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=2,
        torch_module=torch,
    )

    assert generated.tolist() == [[11, 12]]


def test_slice_generated_canvas_keeps_generated_only_canvas() -> None:
    torch = pytest.importorskip("torch")
    token_canvas = torch.tensor([[11, 12, 13]], dtype=torch.long)

    generated = _slice_generated_canvas(
        token_canvas=token_canvas,
        prompt_token_ids=[1, 2, 3, 4],
        max_new_tokens=2,
        torch_module=torch,
    )

    assert generated.tolist() == [[11, 12]]


def test_truncate_generated_ids_drops_prompt_prefix() -> None:
    torch = pytest.importorskip("torch")
    sequence = torch.tensor([1, 2, 3, 11, 12, 0, 13], dtype=torch.long)

    generated = _truncate_generated_ids(
        sequence,
        prompt_token_ids=[1, 2, 3],
        eos_token_id=0,
        pad_token_id=None,
    )

    assert generated == [11, 12]


def test_truncate_generated_ids_keeps_generated_only_sequence() -> None:
    torch = pytest.importorskip("torch")
    sequence = torch.tensor([11, 12, 0, 13], dtype=torch.long)

    generated = _truncate_generated_ids(
        sequence,
        prompt_token_ids=[1, 2, 3],
        eos_token_id=0,
        pad_token_id=None,
    )

    assert generated == [11, 12]


def test_suppress_special_tokens_before_min_new_tokens_only_updates_min_span() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 5, 10), dtype=torch.float32)

    adjusted = _suppress_special_tokens_before_min_new_tokens(
        logits=logits,
        prompt_length=2,
        min_new_tokens=2,
        special_token_ids=(3, 4),
    )

    assert adjusted[0, 2, 3].item() == float("-inf")
    assert adjusted[0, 2, 4].item() == float("-inf")
    assert adjusted[0, 3, 3].item() == float("-inf")
    assert adjusted[0, 4, 3].item() == 0.0
    assert logits[0, 2, 3].item() == 0.0


def test_special_token_ids_flattens_and_deduplicates_values() -> None:
    assert _special_token_ids(1, [2, 1], None, (3,)) == (1, 2, 3)


def test_resolve_trace_profiles_expands_all() -> None:
    profiles = _resolve_trace_profiles(trace_profile="all", alg=None, alg_temp=None)

    assert [profile.name for profile in profiles] == [
        "entropy_det",
        "entropy_stochastic",
        "origin_random",
    ]
    assert [profile.alg for profile in profiles] == ["entropy", "entropy", "origin"]
    assert [profile.alg_temp for profile in profiles] == [0.0, 0.1, None]


def test_resolve_trace_profiles_defaults_to_training_profile() -> None:
    profiles = _resolve_trace_profiles(trace_profile=None, alg=None, alg_temp=None)

    assert len(profiles) == 1
    assert profiles[0].name == "entropy_stochastic"
    assert profiles[0].alg == "entropy"
    assert profiles[0].alg_temp == 0.1


def test_normalize_payload_uses_profile_trace_id() -> None:
    config = DreamGenerationConfig(
        model_name="dream-test",
        trace_profile="entropy_stochastic",
        seed=7,
        alg="entropy",
        alg_temp=0.1,
    )

    normalized = _normalize_payload(
        {
            "steps": [{"step_index": 0, "tokens": [{"token_index": 0, "token_text": "A"}]}],
            "decoding_metadata": {},
        },
        {"sample_id": "sample-1", "prompt": "Prompt", "expected_answer": "A"},
        config,
    )

    assert normalized["trace_id"] == "sample-1__entropy_stochastic__seed-7"
    assert normalized["decoding_metadata"]["trace_profile"] == "entropy_stochastic"
    assert normalized["decoding_metadata"]["temperature"] == config.temperature
    assert normalized["decoding_metadata"]["min_new_tokens"] == config.min_new_tokens
    assert normalized["decoding_metadata"]["expected_answer"] == "A"
    assert normalized["decoding_metadata"]["seed"] == 7


def test_prompt_seed_is_stable_for_same_profile_and_prompt() -> None:
    prompt_record = {"sample_id": "prompt-42", "prompt": "Explain phases."}

    first = _prompt_seed(3, "entropy_stochastic", prompt_record)
    second = _prompt_seed(3, "entropy_stochastic", prompt_record)
    third = _prompt_seed(3, "origin_random", prompt_record)

    assert first == second
    assert first != third


def test_resolve_delimiter_features_skips_multi_token_delimiters() -> None:
    features = _resolve_delimiter_features(_TokenizerStub(), (".", "\n", ".."))

    assert [feature.feature_key for feature in features] == [
        "delimiter_prob_period",
        "delimiter_prob_newline",
    ]


def test_load_dream_model_falls_back_to_device_map_auto_on_cuda_oom() -> None:
    auto_model = _AutoModelStub()

    model = _load_dream_model(
        auto_model=auto_model,
        model_name="dream-test",
        device="cuda",
        dtype="bf16",
        torch_module=_TorchLoadStub(),
    )

    assert model is auto_model.second_model
    assert auto_model.calls[0]["low_cpu_mem_usage"] is True
    assert "device_map" not in auto_model.calls[0]
    assert auto_model.calls[1]["device_map"] == "auto"
    assert auto_model.second_model.eval_called is True


def test_run_dream_trace_dump_skips_empty_generations(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "out"
    prompts_path.write_text(
        "\n".join(
            [
                '{"sample_id":"prompt-001","prompt":"ok"}',
                '{"sample_id":"prompt-002","prompt":"skip"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_collect_trace(prompt_record, config):
        if prompt_record["sample_id"] == "prompt-002":
            raise ValueError(
                "Dream returned no non-special generated tokens. Prompt sample_id='prompt-002'."
            )
        return {
            "trace_id": prompt_record["sample_id"],
            "prompt": prompt_record["prompt"],
            "model_name": config.model_name,
            "decoding_metadata": {},
            "steps": [
                {
                    "step_index": 0,
                    "tokens": [{"token_index": 0, "token_text": "A"}],
                }
            ],
        }

    monkeypatch.setattr(dream_trace_dump, "collect_trace", _fake_collect_trace)
    monkeypatch.setattr(
        dream_trace_dump.sys,
        "argv",
        [
            "run_dream_trace_dump.py",
            "--prompts",
            str(prompts_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = dream_trace_dump.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (output_dir / "prompt-001__entropy_stochastic__seed-0.json").exists()
    assert not (output_dir / "prompt-002__entropy_stochastic__seed-0.json").exists()
    assert "Skipping prompt after empty Dream generation" in captured.err


def test_run_dream_trace_dump_clears_collector_cache_between_profiles(
    tmp_path,
    monkeypatch,
) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "out"
    prompts_path.write_text(
        '{"sample_id":"prompt-001","prompt":"ok"}\n',
        encoding="utf-8",
    )
    clear_calls: list[str] = []

    def _fake_collect_trace(prompt_record, config):
        return {
            "trace_id": prompt_record["sample_id"],
            "prompt": prompt_record["prompt"],
            "model_name": config.model_name,
            "decoding_metadata": {},
            "steps": [
                {
                    "step_index": 0,
                    "tokens": [{"token_index": 0, "token_text": config.trace_profile}],
                }
            ],
        }

    monkeypatch.setattr(dream_trace_dump, "collect_trace", _fake_collect_trace)
    monkeypatch.setattr(
        dream_trace_dump,
        "clear_collector_cache",
        lambda: clear_calls.append("cleared"),
    )
    monkeypatch.setattr(
        dream_trace_dump.sys,
        "argv",
        [
            "run_dream_trace_dump.py",
            "--prompts",
            str(prompts_path),
            "--output-dir",
            str(output_dir),
            "--trace-profile",
            "all",
        ],
    )

    exit_code = dream_trace_dump.main()

    assert exit_code == 0
    assert clear_calls == ["cleared", "cleared", "cleared"]
    assert (output_dir / "prompt-001__entropy_det__seed-0.json").exists()
    assert (output_dir / "prompt-001__entropy_stochastic__seed-0.json").exists()
    assert (output_dir / "prompt-001__origin_random__seed-0.json").exists()
