from __future__ import annotations

from pag.baselines import run_baseline


def test_run_baseline_returns_contract_compliant_outputs(run_config, sample_records) -> None:
    artifacts = run_baseline(run_config, sample_records)

    assert len(artifacts.samples) == len(sample_records)
    assert len(artifacts.requests) == len(sample_records)
    assert len(artifacts.traces) == len(sample_records)
    assert len(artifacts.completions) == len(sample_records)
    assert {trace.sample_id for trace in artifacts.traces} == {
        sample.sample_id for sample in sample_records
    }
    assert all(result.tokens for result in artifacts.completions)
    assert artifacts.summary.stage == "baseline"

