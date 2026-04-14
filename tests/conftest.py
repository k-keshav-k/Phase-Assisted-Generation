from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pag.contracts.schemas import (
    DecodingConfig,
    EvaluationConfig,
    ModelConfig,
    PredictorConfig,
    RunConfig,
    SampleRecord,
    SchedulerConfig,
)


@pytest.fixture
def sample_records() -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id="sample-001",
            prompt="Explain adaptive decoding.",
            reference="It allocates effort across generation phases.",
            metadata={"split": "dev"},
        ),
        SampleRecord(
            sample_id="sample-002",
            prompt="Describe throughput and quality tradeoffs.",
            reference="More refinement can improve quality at higher cost.",
            metadata={"split": "dev"},
        ),
    ]


@pytest.fixture
def run_config(tmp_path: Path, sample_records: list[SampleRecord]) -> RunConfig:
    dataset_path = tmp_path / "samples.yaml"
    dataset_path.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "reference": sample.reference,
                        "metadata": sample.metadata,
                    }
                    for sample in sample_records
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return RunConfig(
        run_id="test-run",
        output_root=str(tmp_path / "artifacts"),
        dataset_path=str(dataset_path),
        seed=123,
        enabled_stages=["baseline", "phases", "scheduler", "evaluation"],
        model=ModelConfig(name="mock-diffusion-lm", revision="test"),
        decoding=DecodingConfig(
            strategy="fixed",
            max_tokens=6,
            chunk_size=2,
            refinement_steps=1,
            temperature=0.0,
        ),
        predictor=PredictorConfig(name="heuristic-phase-predictor", label_space=["easy", "hard"]),
        scheduler=SchedulerConfig(
            name="phase-adaptive-scheduler",
            default_chunk_size=3,
            default_refinement_steps=2,
            parameters={
                "easy_chunk_size": 4,
                "hard_chunk_size": 1,
                "easy_refinement_steps": 1,
                "hard_refinement_steps": 3,
            },
        ),
        evaluation=EvaluationConfig(name="default-eval", metrics=["throughput_proxy"]),
        notes={"suite": "pytest"},
    )

