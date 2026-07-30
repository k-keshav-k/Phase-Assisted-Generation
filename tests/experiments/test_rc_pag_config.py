from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pag.experiments.rc_pag_config import load_rc_pag_config, validate_rc_pag_config

CONFIG_PATH = Path("configs/experiments/rc_pag_neurips.yaml")


def _valid_payload() -> dict:
    return deepcopy(load_rc_pag_config(CONFIG_PATH).raw)


def test_frozen_config_loads_with_declared_family_and_counts():
    config = load_rc_pag_config(CONFIG_PATH)

    assert config.risk.alpha == 0.05
    assert config.risk.delta == 0.05
    assert config.stage_sizes.pilot_per_model == 32
    assert config.stage_sizes.traces_per_model == 600
    assert config.stage_sizes.calibration_per_model == 300
    assert config.statistics.bootstrap_samples == 10_000
    assert len(config.candidates) == 6
    assert {candidate.variant for candidate in config.candidates} == {
        "rc_pag_local",
        "rc_pag_history",
    }
    assert len(config.config_hash) == 64


def test_config_rejects_split_overlap():
    payload = _valid_payload()
    payload["splits"]["calibration"]["gsm8k_train"] = [250, 349]

    with pytest.raises(ValueError, match="overlap"):
        validate_rc_pag_config(payload)


def test_config_rejects_unfrozen_risk_levels():
    payload = _valid_payload()
    payload["risk"]["alpha"] = 0.10

    with pytest.raises(ValueError, match="alpha must remain 0.05"):
        validate_rc_pag_config(payload)


def test_config_rejects_wrong_candidate_family_size():
    payload = _valid_payload()
    payload["policy"]["candidates"].pop()

    with pytest.raises(ValueError, match="exactly six"):
        validate_rc_pag_config(payload)


def test_config_rejects_confirmatory_count_drift():
    payload = _valid_payload()
    payload["confirmatory"]["gsm8k_test"] = 100

    with pytest.raises(ValueError, match="confirmatory counts"):
        validate_rc_pag_config(payload)
