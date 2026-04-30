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

run_eval = importlib.import_module("run_pag_vs_adablock_eval")


class FakeTokenizer:
    def __call__(self, text):
        del text
        return {"input_ids": [1, 2, 3]}

    def decode(self, token_ids, skip_special_tokens=True) -> str:  # noqa: ARG002
        return "".join(chr(96 + int(token_id)) for token_id in token_ids)


def test_load_eval_prompts_reads_expected_substrings(tmp_path) -> None:
    prompt_file = tmp_path / "eval.jsonl"
    prompt_file.write_text(
        '{"id":"math","category":"reasoning","tags":["a"],'
        '"expected_contains":["42"],"expected_answers":["42"],'
        '"prompt":"What is the answer?"}\n',
        encoding="utf-8",
    )

    records = run_eval.load_eval_prompts(prompt_file)

    assert len(records) == 1
    assert records[0].prompt_id == "math"
    assert records[0].category == "reasoning"
    assert records[0].tags == ["a"]
    assert records[0].expected_contains == ["42"]
    assert records[0].expected_answers == ["42"]


def test_substring_score_is_case_insensitive() -> None:
    score = run_eval._substring_score("The answer is Forty Two and 42.", ["42", "answer"])

    assert score == {
        "expected_contains": ["42", "answer"],
        "matched": ["42", "answer"],
        "missing": [],
        "score": 1.0,
    }


def test_answer_score_rejects_embedded_numeric_match() -> None:
    score = run_eval._answer_score("The model produced 772 kilometers.", ["72"])

    assert score == {
        "expected_answers": ["72"],
        "matched": [],
        "missing": ["72"],
        "is_correct": False,
        "score": 0.0,
    }


def test_answer_score_accepts_any_answer_variant() -> None:
    score = run_eval._answer_score("The final price is $97.20.", ["97.2", "97.20"])

    assert score["matched"] == ["97.20"]
    assert score["is_correct"] is True
    assert score["score"] == 1.0


def test_build_history_block_visualization_uses_token_block_boundaries() -> None:
    blocks = run_eval._build_history_block_visualization(
        tokenizer=FakeTokenizer(),
        input_ids=torch.tensor([[99, 98]]),
        output_ids=torch.tensor([[99, 98, 1, 2, 3, 4, 5]]),
        block_history=[2, 3],
        nfe_history=[4, 1],
    )

    assert blocks[0]["block_text"] == "ab"
    assert blocks[0]["applied_block_size"] == 2
    assert blocks[0]["actual_nfe_used"] == 4
    assert blocks[1]["block_text"] == "cde"
    assert blocks[1]["applied_block_size"] == 3
    assert blocks[1]["actual_nfe_used"] == 1


def test_adablock_first_seed_uses_realized_first_tuple() -> None:
    seed = run_eval._adablock_first_seed(
        {
            "block_history": [7, 5],
            "nfe_history": [4, 2],
        }
    )

    assert seed == (7, 4)


def test_args_with_pag_seed_overrides_seed_only() -> None:
    args = run_eval.build_arg_parser().parse_args(
        [
            "--model-path",
            "dummy",
            "--seed-block-length",
            "32",
            "--seed-refinement-steps",
            "4",
        ]
    )

    seeded = run_eval._args_with_pag_seed(
        args,
        seed_block_length=9,
        seed_refinement_steps=3,
    )

    assert seeded.seed_block_length == 9
    assert seeded.seed_refinement_steps == 3
    assert args.seed_block_length == 32
    assert args.seed_refinement_steps == 4
    assert seeded.model_path == args.model_path


def test_seed_from_adablock_flag_defaults_on_and_supports_aliases() -> None:
    parser = run_eval.build_arg_parser()

    default_args = parser.parse_args(["--model-path", "dummy"])
    disabled_args = parser.parse_args(
        ["--model-path", "dummy", "--no-seed-from-adablock-first-block"]
    )
    legacy_disabled_args = parser.parse_args(
        ["--model-path", "dummy", "--no-match-adablock-initial-seed"]
    )

    assert default_args.seed_from_adablock_first_block
    assert not disabled_args.seed_from_adablock_first_block
    assert not legacy_disabled_args.seed_from_adablock_first_block


def test_run_pag_passes_effective_seed_to_scheduler(monkeypatch) -> None:
    captured = {}

    def fake_make_scheduler(args, prompt_text, *, seed):
        captured["seed"] = seed
        captured["prompt_text"] = prompt_text
        return SimpleNamespace(prediction_trace=[{"source": "seed"}])

    def fake_generator(*args, **kwargs):  # noqa: ARG001
        return (
            torch.tensor([[1, 2, 3, 4, 5]]),
            [2],
            [7],
            [
                {
                    "block_index": 0,
                    "predicted_tuple": {"block_size": 7, "refinement_steps": 2},
                    "applied_block_size": 7,
                    "budgeted_refinement_steps": 2,
                    "actual_nfe_used": 2,
                    "block_start": 3,
                    "block_end": 5,
                }
            ],
        )

    monkeypatch.setattr(run_eval, "_make_scheduler", fake_make_scheduler)
    import generate_pag

    monkeypatch.setattr(generate_pag, "generate_pag", fake_generator)
    args = run_eval.build_arg_parser().parse_args(
        [
            "--model-path",
            "dummy",
            "--device",
            "cpu",
            "--seed-block-length",
            "7",
            "--seed-refinement-steps",
            "2",
        ]
    )
    record = run_eval.EvalPromptRecord(prompt="hello", prompt_id="p")

    run_eval._run_pag(args, model=object(), tokenizer=FakeTokenizer(), record=record)

    assert captured["prompt_text"] == "hello"
    assert captured["seed"].block_length == 7
    assert captured["seed"].refinement_steps == 2


def test_comparison_delta_reports_nfe_ratio() -> None:
    pag = {
        "metrics": {
            "total_nfe": 10,
            "elapsed_sec": 2.0,
            "num_blocks": 4,
            "substring_check": {"score": 0.5},
            "answer_check": {"score": 1.0},
        }
    }
    adablock = {
        "metrics": {
            "total_nfe": 20,
            "elapsed_sec": 3.5,
            "num_blocks": 3,
            "substring_check": {"score": 1.0},
            "answer_check": {"score": 0.0},
        }
    }

    delta = run_eval._comparison_delta(pag, adablock)

    assert delta["nfe_delta_pag_minus_adablock"] == -10
    assert delta["nfe_ratio_pag_over_adablock"] == 0.5
    assert delta["elapsed_delta_sec_pag_minus_adablock"] == -1.5
    assert delta["block_count_delta_pag_minus_adablock"] == 1
    assert delta["substring_score_delta_pag_minus_adablock"] == -0.5
    assert delta["answer_score_delta_pag_minus_adablock"] == 1.0
