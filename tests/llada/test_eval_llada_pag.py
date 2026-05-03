from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))


def _install_stubs() -> None:
    lm_eval = ModuleType("lm_eval")
    main = ModuleType("lm_eval.__main__")
    main.cli_evaluate = lambda: None
    api = ModuleType("lm_eval.api")
    api_registry = ModuleType("lm_eval.api.registry")
    api_registry.register_model = lambda *_args, **_kwargs: (lambda cls: cls)

    base_module = ModuleType("eval_llada_adablock")

    class BaseHarness:
        def __init__(self, **kwargs) -> None:
            self.model = kwargs["model"]
            self.tokenizer = kwargs["tokenizer"]
            self.device = kwargs.get("device", "cpu")
            self.rank = 0
            self.mask_id = kwargs.get("mask_id", 126336)
            self.steps = kwargs.get("steps", 6)
            self.gen_length = kwargs.get("gen_length", 2)
            self.block_length = kwargs.get("block_length", 2)
            self.remasking = kwargs.get("remasking", "low_confidence")
            self.use_cache = kwargs.get("use_cache", False)
            self.threshold = kwargs.get("threshold")
            self.save_dir = kwargs.get("save_dir")
            self.show_speed = kwargs.get("show_speed", False)
            self.dual_cache = kwargs.get("dual_cache", False)
            self.is_instruct = kwargs.get("is_instruct", False)

    base_module.LLaDAEvalHarness = BaseHarness

    sys.modules["lm_eval"] = lm_eval
    sys.modules["lm_eval.__main__"] = main
    sys.modules["lm_eval.api"] = api
    sys.modules["lm_eval.api.registry"] = api_registry
    sys.modules["eval_llada_adablock"] = base_module


def test_eval_llada_pag_wires_predictor_args_into_generation(monkeypatch) -> None:
    _install_stubs()
    eval_llada_pag = importlib.import_module("eval_llada_pag")

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            self.init_kwargs = kwargs
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeTokenizer:
        def __call__(self, text):
            del text
            return {"input_ids": [1, 2]}

        def decode(self, token_ids, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            return "decoded"

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            del messages, add_generation_prompt, tokenize
            return "templated"

    class FakeRequest:
        def __init__(self) -> None:
            self.args = ("question", {"until": ["stop"]})
            self.doc = {}

    calls: list[dict[str, object]] = []

    def fake_generate_pag(*args, **kwargs):
        del args
        calls.append(kwargs)
        return (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            [2],
            [2],
            [{"block_index": 0}],
        )

    monkeypatch.setattr(eval_llada_pag, "PAGTupleScheduler", FakeScheduler)
    monkeypatch.setattr(eval_llada_pag, "generate_pag", fake_generate_pag)

    model = SimpleNamespace()
    tokenizer = FakeTokenizer()
    harness = eval_llada_pag.LLaDAEvalHarness(
        model_path="dummy",
        batch_size=1,
        steps=6,
        gen_length=2,
        block_length=4,
        threshold=0.85,
        predictor_ckpt="checkpoint.pt",
        seed_block_length=3,
        seed_refinement_steps=2,
        predictor_device="cpu",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    output = harness.generate_until([FakeRequest()])

    assert harness.max_block_length == 4
    assert harness.max_refinement_steps == 6
    assert harness.pag_scheduler.init_kwargs == {
        "predictor_ckpt": "checkpoint.pt",
        "seed_block_length": 3,
        "seed_refinement_steps": 2,
        "predictor_device": "cpu",
        "context_seed_block_length": None,
        "context_seed_stabilizing_steps": None,
        "min_refinement_steps": 3,
    }
    assert harness.schedule_histories == [[{"block_index": 0}]]
    assert len(calls) == 1
    assert calls[0] == {
        "steps": 6,
        "gen_length": 2,
        "temperature": 0.0,
        "remasking": "low_confidence",
        "mask_id": 126336,
        "threshold": 0.85,
        "max_block_length": 4,
        "max_refinement_steps": 6,
    }
    assert output == ["decoded"]
