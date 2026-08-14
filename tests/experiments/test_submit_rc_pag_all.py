from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/slurm/submit_rc_pag_all.sh")
WORKER = Path("scripts/slurm/rc_pag_a100.sbatch")
RUNBOOK = Path("docs/rc_pag_one_command.md")


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "sbatch.txt"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        """#!/bin/bash
set -euo pipefail
{
    printf 'stage=%s\\n' "$RC_PAG_STAGE"
    printf 'confirm=%s\\n' "$RC_PAG_ALLOW_CONFIRMATORY"
    printf 'config=%s\\n' "$RC_PAG_CONFIG"
    printf 'reuse=%s\\n' "${RC_PAG_REUSE_FROM:-}"
    printf 'args='
    printf '%s ' "$@"
    printf '\\n'
} > "$RC_PAG_CAPTURE"
printf 'Submitted batch job 12345\\n'
""",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    overlay = tmp_path / "overlay.ext3"
    image = tmp_path / "image.sif"
    overlay.touch()
    image.touch()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_DIR": str(Path.cwd()),
            "RC_PAG_OUTPUT_ROOT": str(tmp_path / "artifacts"),
            "OVERLAY_PATH": str(overlay),
            "SIF_PATH": str(image),
            "RC_PAG_CAPTURE": str(capture),
        }
    )
    return env, capture


def test_one_command_submits_all_stages_with_confirmation(tmp_path):
    env, capture = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    submitted = capture.read_text(encoding="utf-8")
    assert "stage=all" in submitted
    assert "confirm=1" in submitted
    assert "config=" in submitted
    assert "rc_pag_neurips_workshop_v9.yaml" in submitted
    assert "--time=48:00:00" in submitted
    assert "rc_pag_a100.sbatch" in submitted
    assert "32-prompt/model numerical audit" in result.stdout
    assert "64-prompt/model gate" in result.stdout
    assert ">5% paired-bootstrap latency-reduction" in result.stdout
    assert "rollout" not in result.stdout
    assert "refit" not in result.stdout


def test_one_command_forwards_compatible_reuse_source(tmp_path):
    env, capture = _environment(tmp_path)
    source = tmp_path / "completed-v5"
    source.mkdir()
    env["RC_PAG_REUSE_FROM"] = str(source)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"reuse={source}" in capture.read_text(encoding="utf-8")


def test_one_command_rejects_development_limit(tmp_path):
    env, capture = _environment(tmp_path)
    env["RC_PAG_LIMIT"] = "2"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "RC_PAG_LIMIT" in result.stderr
    assert not capture.exists()


def test_a100_worker_bootstraps_hf_assets_before_experiment() -> None:
    worker = WORKER.read_text(encoding="utf-8")

    cuda_check = "python scripts/check_rc_pag_cuda.py"
    bootstrap = "python scripts/bootstrap_rc_pag_hf.py"
    experiment = "python scripts/run_rc_pag.py"
    assert 'RC_PAG_HF_MODE="${RC_PAG_HF_MODE:-auto}"' in worker
    assert "RC_PAG_HF_OFFLINE:-" in worker
    assert bootstrap in worker
    assert '--config "$RC_PAG_CONFIG"' in worker
    assert '--mode "$RC_PAG_HF_MODE"' in worker
    assert cuda_check in worker
    assert worker.index(cuda_check) < worker.index(bootstrap)
    assert worker.index(bootstrap) < worker.index(experiment)


def test_a100_worker_requeues_retryable_cuda_allocations_with_a_cap() -> None:
    worker = WORKER.read_text(encoding="utf-8")

    assert 'RC_PAG_MAX_CUDA_REQUEUES="${RC_PAG_MAX_CUDA_REQUEUES:-2}"' in worker
    assert "SLURM_RESTART_COUNT:-0" in worker
    assert 'if [[ "${status}" -eq 75 ]]' in worker
    assert 'scontrol requeue "${SLURM_JOB_ID}"' in worker
    assert "CUDA allocation smoke check failed" in worker
    assert "automatic CUDA requeue limit" in worker


def test_a100_worker_runs_auto_and_offline_modes_offline_after_bootstrap() -> None:
    worker = WORKER.read_text(encoding="utf-8")

    assert 'if [[ "$RC_PAG_HF_MODE" == "online" ]]' in worker
    assert "export HF_HUB_OFFLINE=1" in worker
    assert "export HF_DATASETS_OFFLINE=1" in worker
    assert "export TRANSFORMERS_OFFLINE=1" in worker
    assert "export HF_HUB_OFFLINE=0" in worker
    assert "export HF_DATASETS_OFFLINE=0" in worker
    assert "export TRANSFORMERS_OFFLINE=0" in worker


def test_one_command_runbook_starts_with_the_single_submission_command() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    assert "bash scripts/slurm/submit_rc_pag_all.sh" in runbook.split("```bash", 1)[1]
    assert "rc-pag-588204cfb482" in runbook
    assert "8--15 A100" in runbook
