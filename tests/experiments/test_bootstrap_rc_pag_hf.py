from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from pag.experiments.rc_pag_config import load_rc_pag_config


def _load_bootstrap_module():
    path = Path("scripts/bootstrap_rc_pag_hf.py").resolve()
    spec = importlib.util.spec_from_file_location("bootstrap_rc_pag_hf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bootstrap():
    return _load_bootstrap_module()


@pytest.fixture
def v8_config():
    return load_rc_pag_config(Path("configs/experiments/rc_pag_neurips_workshop_v8.yaml"))


def test_discover_assets_includes_all_pinned_models_and_dataset_splits(
    bootstrap, v8_config
) -> None:
    assets = bootstrap.discover_assets(v8_config)

    assert len(assets) == 15
    assert {(asset.repository, asset.revision) for asset in assets if asset.kind == "model"} == {
        (spec.repository, spec.revision) for spec in v8_config.models.values()
    }
    dataset_assets = {
        (asset.repository, asset.config, asset.split, asset.revision)
        for asset in assets
        if asset.kind == "dataset"
    }
    assert (
        "openai/gsm8k",
        "main",
        "train",
        v8_config.datasets["gsm8k_train"].revision,
    ) in dataset_assets
    assert (
        "openai/gsm8k",
        "main",
        "test",
        v8_config.datasets["gsm8k_test"].revision,
    ) in dataset_assets
    assert len(dataset_assets) == 13


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "auto"),
        ({"RC_PAG_HF_OFFLINE": "1"}, "offline"),
        ({"RC_PAG_HF_OFFLINE": "true"}, "offline"),
        ({"RC_PAG_HF_OFFLINE": "0"}, "online"),
        ({"RC_PAG_HF_MODE": "offline", "RC_PAG_HF_OFFLINE": "0"}, "offline"),
        ({"RC_PAG_HF_MODE": "online", "RC_PAG_HF_OFFLINE": "1"}, "online"),
    ],
)
def test_resolve_mode_precedence(bootstrap, environment, expected) -> None:
    assert bootstrap.resolve_mode(environment) == expected


@pytest.mark.parametrize(
    "environment",
    [
        {"RC_PAG_HF_MODE": "sometimes"},
        {"RC_PAG_HF_OFFLINE": "sometimes"},
    ],
)
def test_resolve_mode_rejects_invalid_values(bootstrap, environment) -> None:
    with pytest.raises(ValueError, match="RC_PAG_HF"):
        bootstrap.resolve_mode(environment)


def _assets(bootstrap):
    return (
        bootstrap.HFAsset("model", "llada", "org/model", "a" * 40),
        bootstrap.HFAsset(
            "dataset",
            "gsm8k:main:train",
            "org/data",
            "b" * 40,
            config="main",
            split="train",
        ),
    )


def test_auto_mode_fetches_only_missing_assets_then_verifies(bootstrap) -> None:
    all_assets = _assets(bootstrap)
    missing = (all_assets[1],)
    calls = []
    outcomes = [missing, (), ()]

    def runner(action, assets, *, offline):
        calls.append((action, tuple(assets), offline))
        return outcomes.pop(0)

    bootstrap.bootstrap_assets(all_assets, mode="auto", worker_runner=runner)

    assert calls == [
        ("probe", all_assets, True),
        ("fetch", missing, False),
        ("probe", all_assets, True),
    ]


def test_auto_mode_with_complete_cache_never_fetches(bootstrap) -> None:
    all_assets = _assets(bootstrap)
    calls = []

    def runner(action, assets, *, offline):
        calls.append((action, tuple(assets), offline))
        return ()

    bootstrap.bootstrap_assets(all_assets, mode="auto", worker_runner=runner)

    assert calls == [("probe", all_assets, True)]


def test_offline_mode_never_fetches_and_reports_missing_assets(bootstrap) -> None:
    all_assets = _assets(bootstrap)
    calls = []

    def runner(action, assets, *, offline):
        calls.append((action, tuple(assets), offline))
        return (all_assets[0],)

    with pytest.raises(bootstrap.BootstrapError, match="model:llada"):
        bootstrap.bootstrap_assets(all_assets, mode="offline", worker_runner=runner)

    assert calls == [("probe", all_assets, True)]


def test_online_mode_fetches_all_assets_once(bootstrap) -> None:
    all_assets = _assets(bootstrap)
    calls = []

    def runner(action, assets, *, offline):
        calls.append((action, tuple(assets), offline))
        return ()

    bootstrap.bootstrap_assets(all_assets, mode="online", worker_runner=runner)

    assert calls == [("fetch", all_assets, False)]


def test_auto_mode_requires_final_offline_verification(bootstrap) -> None:
    all_assets = _assets(bootstrap)
    calls = []
    outcomes = [(all_assets[0],), (), (all_assets[0],)]

    def runner(action, assets, *, offline):
        calls.append((action, tuple(assets), offline))
        return outcomes.pop(0)

    with pytest.raises(bootstrap.BootstrapError, match="model:llada"):
        bootstrap.bootstrap_assets(all_assets, mode="auto", worker_runner=runner)

    assert calls[-1] == ("probe", all_assets, True)


def test_worker_environment_controls_all_offline_flags(bootstrap, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    online = bootstrap.worker_environment(offline=False)
    offline = bootstrap.worker_environment(offline=True)

    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert online[name] == "0"
        assert offline[name] == "1"
