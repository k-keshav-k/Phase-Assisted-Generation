from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_DIR = REPO_ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

run_pag_dummy_api = importlib.import_module("run_pag_dummy_api")


class FakePredictor:
    def __init__(self, outputs, window_size: int = 4) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[run_pag_dummy_api.BlockTuple]] = []
        self.config = SimpleNamespace(window_size=window_size)

    def predict(self, context):
        self.calls.append(list(context))
        predicted_tuple = self.outputs[len(self.calls) - 1]
        return SimpleNamespace(
            predicted_tuple=predicted_tuple,
            raw_output=[
                float(predicted_tuple.block_size),
                float(predicted_tuple.refinement_steps),
            ],
            metadata={"window_size_used": len(context)},
        )


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "|".join(str(token_id) for token_id in token_ids)


def test_parse_tuple_schedule_parses_entries() -> None:
    parsed = run_pag_dummy_api.parse_tuple_schedule("16:4,8:2,4:1")

    assert parsed == [
        run_pag_dummy_api.BlockTuple(16, 4),
        run_pag_dummy_api.BlockTuple(8, 2),
        run_pag_dummy_api.BlockTuple(4, 1),
    ]


def test_load_prompt_records_from_jsonl(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"a","category":"math","tags":["short"],"prompt":"What is 2+2?"}\n'
        '{"id":"b","category":"code","prompt":"Write a loop."}\n',
        encoding="utf-8",
    )

    records = run_pag_dummy_api.load_prompt_records(prompt_file)

    assert [record.prompt_id for record in records] == ["a", "b"]
    assert records[0].category == "math"
    assert records[0].tags == ["short"]
    assert records[1].prompt == "Write a loop."


def test_write_log_record_appends_jsonl(tmp_path) -> None:
    log_file = tmp_path / "logs" / "runs.jsonl"
    run_pag_dummy_api.write_log_record(log_file, {"prompt_id": "a", "value": 1})
    run_pag_dummy_api.write_log_record(log_file, {"prompt_id": "b", "value": 2})

    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]

    assert rows == [{"prompt_id": "a", "value": 1}, {"prompt_id": "b", "value": 2}]


def test_dummy_scheduler_uses_seed_then_dummy_api_and_clamps() -> None:
    api = run_pag_dummy_api.DummyTupleAPI(
        scripted_tuples=[run_pag_dummy_api.BlockTuple(9, 0)],
        fallback_block_size=5,
        fallback_refinement_steps=3,
        verbose=False,
    )
    scheduler = run_pag_dummy_api.DummyAPIScheduler(
        prompt_text="test prompt",
        seed_block_length=6,
        seed_refinement_steps=2,
        api=api,
    )

    first = scheduler.next_schedule(
        remaining_tokens=20,
        max_block_length=10,
        max_refinement_steps=8,
    )
    scheduler.record_realized(first.applied_block_size, 4)

    second = scheduler.next_schedule(
        remaining_tokens=3,
        max_block_length=2,
        max_refinement_steps=7,
    )

    assert first.predicted_tuple == run_pag_dummy_api.BlockTuple(6, 2)
    assert second.predicted_tuple == run_pag_dummy_api.BlockTuple(9, 0)
    assert second.applied_block_size == 2
    assert second.budgeted_refinement_steps == 1
    assert api.requests == [
        {
            "prompt": "test prompt",
            "block_index": 1,
            "remaining_tokens": 3,
            "history": [{"block_size": 6, "refinement_steps": 4}],
        }
    ]
    assert scheduler.prediction_trace[1]["source"] == "dummy_api"


def test_max_stabilizing_step_matches_trace_style_max() -> None:
    predictions = [
        torch.tensor([4, 9, 1]),
        torch.tensor([4, 8, 1]),
        torch.tensor([4, 8, 2]),
        torch.tensor([4, 8, 2]),
    ]

    assert (
        run_pag_dummy_api._max_stabilizing_step(
            predictions,
            torch.tensor([4, 8, 2]),
        )
        == 2
    )


def test_effective_seed_uses_explicit_seed_when_adablock_probe_disabled() -> None:
    args = run_pag_dummy_api.build_arg_parser().parse_args(
        [
            "--model-path",
            "dummy",
            "--prompt",
            "hello",
            "--seed-block-length",
            "12",
            "--seed-refinement-steps",
            "5",
            "--no-seed-from-adablock-first-block",
        ]
    )

    seed = run_pag_dummy_api._effective_seed(args=args, model=None, input_ids=None)

    assert seed == run_pag_dummy_api.EffectiveSeed(
        block_length=12,
        refinement_steps=5,
        source="explicit",
        context_stabilizing_steps=4,
    )


def test_make_scheduler_uses_effective_seed_for_block_zero() -> None:
    args = run_pag_dummy_api.build_arg_parser().parse_args(
        [
            "--model-path",
            "dummy",
            "--prompt",
            "hello",
            "--dummy-tuples",
            "3:1",
        ]
    )
    seed = run_pag_dummy_api.EffectiveSeed(
        block_length=9,
        refinement_steps=4,
        source="adablock_first_block",
    )

    scheduler = run_pag_dummy_api._make_scheduler(args, "hello", seed=seed)
    first = scheduler.next_schedule(
        remaining_tokens=32,
        max_block_length=32,
        max_refinement_steps=8,
    )

    assert first.predicted_tuple == run_pag_dummy_api.BlockTuple(9, 4)


def test_seed_from_adablock_first_block_defaults_on_and_can_be_disabled() -> None:
    parser = run_pag_dummy_api.build_arg_parser()

    default_args = parser.parse_args(["--model-path", "dummy", "--prompt", "hello"])
    assert default_args.seed_from_adablock_first_block
    assert not parser.parse_args(
        [
            "--model-path",
            "dummy",
            "--prompt",
            "hello",
            "--no-seed-from-adablock-first-block",
        ]
    ).seed_from_adablock_first_block


def test_checkpoint_scheduler_uses_seed_then_predictor_context() -> None:
    predictor = FakePredictor([run_pag_dummy_api.BlockTuple(5, 3)])
    scheduler = run_pag_dummy_api.CheckpointTupleScheduler(
        prompt_text="hello",
        predictor_ckpt="unused.pt",
        seed_block_length=8,
        seed_refinement_steps=2,
        predictor=predictor,
    )

    first = scheduler.next_schedule(
        remaining_tokens=32,
        max_block_length=16,
        max_refinement_steps=12,
    )
    scheduler.record_realized(first.applied_block_size, 4)

    second = scheduler.next_schedule(
        remaining_tokens=24,
        max_block_length=16,
        max_refinement_steps=12,
    )
    scheduler.record_realized(second.applied_block_size, second.budgeted_refinement_steps)

    assert first.predicted_tuple == run_pag_dummy_api.BlockTuple(8, 2)
    assert second.predicted_tuple == run_pag_dummy_api.BlockTuple(5, 4)
    assert predictor.calls == [
        [
            run_pag_dummy_api.BlockTuple(8, 1),
            run_pag_dummy_api.BlockTuple(8, 1),
            run_pag_dummy_api.BlockTuple(8, 1),
            run_pag_dummy_api.BlockTuple(8, 3),
        ]
    ]
    assert scheduler.prediction_trace[1]["source"] == "checkpoint"
    assert scheduler.prediction_trace[1]["raw_output"] == [5.0, 3.0]
    assert scheduler.prediction_trace[1]["metadata"]["stabilizing_step_offset"] == 1
    assert scheduler.prediction_trace[1]["realized_tuple"] == {
        "block_size": 5,
        "refinement_steps": 3,
    }
    assert scheduler.prediction_trace[1]["realized_decode_tuple"] == {
        "block_size": 5,
        "refinement_steps": 4,
    }


def test_build_block_visualization_includes_block_text() -> None:
    tokenizer = FakeTokenizer()
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)
    output_ids = torch.tensor([[10, 11, 21, 22, 23, 24]], dtype=torch.long)
    schedule_history = [
        {
            "block_index": 0,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 3},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 3,
            "actual_nfe_used": 2,
            "block_start": 2,
            "block_end": 4,
        },
        {
            "block_index": 1,
            "predicted_tuple": {"block_size": 2, "refinement_steps": 2},
            "applied_block_size": 2,
            "budgeted_refinement_steps": 2,
            "actual_nfe_used": 1,
            "block_start": 4,
            "block_end": 6,
        },
    ]
    prediction_trace = [
        {"source": "seed"},
        {"source": "checkpoint"},
    ]

    blocks = run_pag_dummy_api._build_block_visualization(
        tokenizer=tokenizer,
        input_ids=input_ids,
        output_ids=output_ids,
        schedule_history=schedule_history,
        prediction_trace=prediction_trace,
    )

    assert blocks[0]["block_text"] == "21|22"
    assert blocks[0]["text_so_far"] == "21|22"
    assert blocks[1]["block_text"] == "23|24"
    assert blocks[1]["text_so_far"] == "21|22|23|24"
    assert blocks[1]["predictor_trace"] == {"source": "checkpoint"}


def test_disable_torch_compile_replaces_compile_with_identity() -> None:
    original_compile = torch.compile
    try:
        run_pag_dummy_api._maybe_disable_torch_compile(True)

        @torch.compile()
        def identity(value):
            return value + 1

        assert identity(2) == 3
        assert getattr(torch.compile, "__name__", "") == "_identity_torch_compile"
    finally:
        torch.compile = original_compile
