from __future__ import annotations

from pag.experiments.cross_model_orchestrator import run_missing_records, select_joint_policy
from pag.experiments.datasets import ExperimentSample
from pag.experiments.records import RecordStore


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(self, sample, method, *, baseline=None, policy=None):
        del baseline, policy
        self.calls.append((method, sample.sample_id))
        return {
            "method": method,
            "sample_id": sample.sample_id,
            "grade": {"is_correct": True},
            "total_nfe": 8 if method == "residual_pag" else 10,
            "nfe_history": [8],
            "block_history": [8],
            "schedule_history": [{}],
            "elapsed_sec": 1.0,
        }


def test_resume_runs_only_missing_keys(tmp_path) -> None:
    store = RecordStore(tmp_path, {"config_hash": "abc"})
    sample = ExperimentSample("gsm8k_train_6300", "gsm8k", "prompt", "1")
    store.write(
        "test_gsm8k/llada",
        "adablock",
        sample.sample_id,
        {
            "grade": {"is_correct": True},
            "total_nfe": 10,
            "nfe_history": [10],
            "block_history": [8],
            "schedule_history": [{}],
            "elapsed_sec": 1.0,
        },
    )
    runtime = FakeRuntime()
    run_missing_records(
        store=store,
        runtime=runtime,
        stage="test_gsm8k/llada",
        samples=[sample],
        methods=("adablock", "residual_pag"),
    )
    assert runtime.calls == [("residual_pag", sample.sample_id)]


def _rows(method: str, correct: int, nfe: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index}",
            "method": method,
            "grade": {"is_correct": index < correct},
            "total_nfe": nfe,
        }
        for index in range(10)
    ]


def test_joint_selection_rejects_low_accuracy_candidate() -> None:
    records = {
        "llada": {
            "adablock": _rows("adablock", 9, 10),
            "residual_q15_c3": _rows("residual_q15_c3", 6, 5),
            "residual_q25_c2": _rows("residual_q25_c2", 9, 8),
        },
        "dream": {
            "adablock": _rows("adablock", 9, 10),
            "residual_q15_c3": _rows("residual_q15_c3", 7, 5),
            "residual_q25_c2": _rows("residual_q25_c2", 9, 7),
        },
    }
    result = select_joint_policy(records, max_correct_loss=2)
    assert result["selected"] == "residual_q25_c2"
    assert not result["fallback"]
