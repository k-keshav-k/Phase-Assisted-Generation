from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/slurm/submit_rc_pag_all.sh")


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
    assert "rc_pag_neurips_workshop_v4.yaml" in submitted
    assert "--time=48:00:00" in submitted
    assert "rc_pag_a100.sbatch" in submitted


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
