from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DREAM_DIR = REPO_ROOT / "AdaBlock-dLLM" / "dream"
if str(DREAM_DIR) not in sys.path:
    sys.path.insert(0, str(DREAM_DIR))


def _install_lm_eval_stubs() -> None:
    lm_eval = ModuleType("lm_eval")
    utils = ModuleType("lm_eval.utils")
    utils.simple_parse_args_string = lambda _: {}

    api = ModuleType("lm_eval.api")
    api_instance = ModuleType("lm_eval.api.instance")
    api_instance.Instance = object
    api_model = ModuleType("lm_eval.api.model")
    api_model.LM = object
    api_registry = ModuleType("lm_eval.api.registry")
    api_registry.register_model = lambda *_args, **_kwargs: lambda cls: cls

    models = ModuleType("lm_eval.models")
    models_utils = ModuleType("lm_eval.models.utils")
    models_utils.get_dtype = lambda _: torch.float32

    main = ModuleType("lm_eval.__main__")
    main.cli_evaluate = lambda: None

    sys.modules.setdefault("lm_eval", lm_eval)
    sys.modules.setdefault("lm_eval.utils", utils)
    sys.modules.setdefault("lm_eval.api", api)
    sys.modules.setdefault("lm_eval.api.instance", api_instance)
    sys.modules.setdefault("lm_eval.api.model", api_model)
    sys.modules.setdefault("lm_eval.api.registry", api_registry)
    sys.modules.setdefault("lm_eval.models", models)
    sys.modules.setdefault("lm_eval.models.utils", models_utils)
    sys.modules.setdefault("lm_eval.__main__", main)


def _install_dream_dependency_stubs() -> None:
    transformers = ModuleType("transformers")
    transformers.__version__ = "0.0"
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
    transformers.PreTrainedModel = object

    generation = ModuleType("transformers.generation")
    generation_configuration = ModuleType("transformers.generation.configuration_utils")

    class GenerationConfig:
        pass

    generation_configuration.GenerationConfig = GenerationConfig

    utils = ModuleType("transformers.utils")
    utils.ModelOutput = object
    utils.is_torchdynamo_compiling = lambda: False
    utils.logging = SimpleNamespace(get_logger=logging.getLogger)

    accelerate = ModuleType("accelerate")

    class Accelerator:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.num_processes = 1
            self.device = torch.device("cpu")

    class InitProcessGroupKwargs:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    accelerate.Accelerator = Accelerator
    accelerate.InitProcessGroupKwargs = InitProcessGroupKwargs

    datasets = ModuleType("datasets")
    datasets.Dataset = object

    model_pkg = ModuleType("model")
    model_pkg.__path__ = [str(DREAM_DIR / "model")]

    generation_utils_adablock = ModuleType("model.generation_utils_adablock")
    generation_utils_adablock.DreamGenerationMixin = type("DreamGenerationMixin", (), {})

    generation_utils_pag = ModuleType("model.generation_utils_pag")
    generation_utils_pag.DreamGenerationMixin = type("DreamGenerationMixin", (), {})

    configuration_dream = ModuleType("model.configuration_dream")
    configuration_dream.DreamConfig = object

    modeling_dream = ModuleType("model.modeling_dream")
    modeling_dream.DreamModel = type("DreamModel", (), {})

    sys.modules["transformers"] = transformers
    sys.modules["transformers.generation"] = generation
    sys.modules["transformers.generation.configuration_utils"] = generation_configuration
    sys.modules["transformers.utils"] = utils
    sys.modules["accelerate"] = accelerate
    sys.modules["datasets"] = datasets
    sys.modules["model"] = model_pkg
    sys.modules["model.generation_utils_adablock"] = generation_utils_adablock
    sys.modules["model.generation_utils_pag"] = generation_utils_pag
    sys.modules["model.configuration_dream"] = configuration_dream
    sys.modules["model.modeling_dream"] = modeling_dream


def test_eval_dream_pag_wires_predictor_args_into_generation(monkeypatch) -> None:
    _install_lm_eval_stubs()
    _install_dream_dependency_stubs()
    eval_dream_pag = importlib.import_module("eval_dream_pag")

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            self.init_kwargs = kwargs
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        eos_token = "<eos>"
        bos_token = "<bos>"

        def __call__(self, prompts, return_tensors=None, padding=None, padding_side=None):
            del prompts, return_tensors, padding, padding_side
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

        def decode(self, token_ids, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            return "decoded"

    class FakeModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.calls: list[dict[str, object]] = []
            self.pag_scheduler = None

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def diffusion_generate(self, prompt_ids, **kwargs):
            self.calls.append(kwargs)
            generated = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            return SimpleNamespace(
                sequences=generated,
                nfe_history=[2],
                block_history=[2],
                schedule_history=[{"block_index": 0}],
            )

    def fake_create(self, pretrained, dtype, trust_remote_code):
        del pretrained, dtype, trust_remote_code
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()

    monkeypatch.setattr(eval_dream_pag, "PAGTupleScheduler", FakeScheduler)
    monkeypatch.setattr(eval_dream_pag.Dream, "_create_model_and_tokenizer", fake_create)

    dream = eval_dream_pag.Dream(
        pretrained="dummy-model",
        device="cpu",
        max_new_tokens=2,
        diffusion_steps=6,
        block_length=4,
        threshold=0.85,
        use_cache=True,
        dual_cache=True,
        predictor_ckpt="checkpoint.pt",
        seed_block_length=3,
        seed_refinement_steps=2,
    )

    responses, nfe_history, block_history, schedule_history = dream._generate_batch(["prompt"])

    assert dream.max_block_length == 4
    assert dream.max_refinement_steps == 6
    assert dream.pag_scheduler.init_kwargs == {
        "predictor_ckpt": "checkpoint.pt",
        "seed_block_length": 3,
        "seed_refinement_steps": 2,
        "predictor_device": "cpu",
        "context_seed_block_length": None,
        "context_seed_stabilizing_steps": None,
        "min_refinement_steps": 3,
    }
    assert dream.pag_scheduler.reset_calls == 1
    assert dream.model.pag_scheduler is dream.pag_scheduler
    assert len(dream.model.calls) == 1
    call = dream.model.calls[0]
    assert torch.equal(call["attention_mask"], torch.tensor([[True, True]]))
    assert call["max_new_tokens"] == 2
    assert call["output_history"] is False
    assert call["return_dict_in_generate"] is True
    assert call["steps"] == 6
    assert call["temperature"] == 0.0
    assert call["top_p"] is None
    assert call["top_k"] is None
    assert call["alg"] == "confidence_threshold"
    assert call["alg_temp"] == 0.0
    assert call["threshold"] == 0.85
    assert call["dual_cache"] is True
    assert call["block_length"] == 4
    assert call["max_block_length"] == 4
    assert call["max_refinement_steps"] == 6
    assert responses == ["decoded"]
    assert nfe_history == [2]
    assert block_history == [2]
    assert schedule_history == [{"block_index": 0}]
