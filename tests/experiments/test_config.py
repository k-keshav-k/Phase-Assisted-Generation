from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from pag.experiments.config import load_experiment_config, validate_experiment_config

CONFIG = Path("configs/experiments/neurips_strategy1.yaml")


def test_loads_frozen_protocol_and_hashes_it() -> None:
    config = load_experiment_config(CONFIG)
    assert config.schema_version == 1
    assert config.seed == 20260710
    assert len(config.config_hash) == 64
    assert len(config.methods.development) == 8


def test_rejects_nonzero_temperature() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["decoding"]["temperature"] = 0.1
    with pytest.raises(ValueError, match="temperature must be 0"):
        validate_experiment_config(payload)
