from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/run_rc_pag.py")


def test_cli_mock_preflight(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "preflight",
            "--mock",
            "--output-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Preflight complete" in result.stdout
    assert "Config hash:" in result.stdout
    assert "Resume command:" in result.stdout


def test_cli_requires_explicit_confirmation_flag(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "confirm",
            "--output-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--allow-confirmatory" in result.stderr


def test_cli_rejects_confirmation_limit_even_with_flag(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "confirm",
            "--output-root",
            str(tmp_path),
            "--allow-confirmatory",
            "--limit",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "rejects --limit" in result.stderr
