from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pag.experiments.rc_pag_runtime import (
    _ensure_llada_config_compatibility,
    _import_llada_model_class_without_compile,
    prompt_loss_from_schedules,
    training_examples_from_schedules,
)


def _schedule():
    observation = {
        "step_index": 1,
        "block_size": 2,
        "masked": [True, True],
        "top1_probs": [0.8, 0.7],
        "top2_probs": [0.1, 0.2],
        "entropies": [0.5, 0.6],
        "token_ids": [5, 6],
        "digit_ids": [5],
        "delimiter_ids": [6],
    }
    return [
        {
            "applied_block_size": 2,
            "actual_nfe_used": 3,
            "final_tokens": [5, 7],
            "risk_steps": [
                {
                    "observation": observation,
                    "proposed_tokens": [5, 6],
                    "shadow_loss": 1,
                }
            ],
            "shadow_losses": [1],
        }
    ]


def test_training_examples_label_proposals_against_full_trajectory_final_tokens():
    examples = training_examples_from_schedules(_schedule(), history_window=4)

    assert len(examples) == 1
    assert examples[0]["unsafe"]
    assert examples[0]["features"]["local.step_index"] == 1.0
    assert examples[0]["features"]["history.length"] == 0.0


def test_prompt_loss_is_any_on_policy_shadow_disagreement():
    assert prompt_loss_from_schedules(_schedule()) == 1
    schedule = _schedule()
    schedule[0]["shadow_losses"] = [0]
    assert prompt_loss_from_schedules(schedule) == 0


def test_llada_config_compatibility_aliases_missing_training_length() -> None:
    config = SimpleNamespace(max_sequence_length=4096)

    result = _ensure_llada_config_compatibility(config)

    assert result is config
    assert config.train_max_sequence_length == 4096


def test_llada_config_compatibility_preserves_checkpoint_value() -> None:
    config = SimpleNamespace(max_sequence_length=4096, train_max_sequence_length=2048)

    _ensure_llada_config_compatibility(config)

    assert config.train_max_sequence_length == 2048


def test_llada_model_import_disables_compile_and_restores_torch(monkeypatch) -> None:
    original_compile = torch.compile
    observed: dict[str, object] = {}
    sentinel = object()

    def fake_import(name: str):
        observed["module"] = name
        observed["compile_name"] = torch.compile.__name__

        @torch.compile()
        def identity(value):
            return value

        observed["identity"] = identity(3)
        return SimpleNamespace(LLaDAModelLM=sentinel)

    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.importlib.import_module",
        fake_import,
    )

    result = _import_llada_model_class_without_compile()

    assert result is sentinel
    assert observed == {
        "module": "model.modeling_llada",
        "compile_name": "_identity_torch_compile",
        "identity": 3,
    }
    assert torch.compile is original_compile


def test_llada_model_import_restores_compile_after_import_error(monkeypatch) -> None:
    original_compile = torch.compile

    def fail_import(name: str):
        assert name == "model.modeling_llada"
        assert torch.compile.__name__ == "_identity_torch_compile"
        raise RuntimeError("import failed")

    monkeypatch.setattr(
        "pag.experiments.rc_pag_runtime.importlib.import_module",
        fail_import,
    )

    with pytest.raises(RuntimeError, match="import failed"):
        _import_llada_model_class_without_compile()

    assert torch.compile is original_compile
