from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pag.experiments.rc_pag_config import RCPAGConfig, load_rc_pag_config

OFFLINE_VARIABLES = (
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
VALID_MODES = {"auto", "offline", "online"}


class BootstrapError(RuntimeError):
    """Raised when pinned Hugging Face assets cannot be prepared."""


@dataclass(frozen=True, slots=True)
class HFAsset:
    kind: str
    name: str
    repository: str
    revision: str
    config: str | None = None
    split: str | None = None

    @property
    def identity(self) -> str:
        return f"{self.kind}:{self.name} ({self.repository}@{self.revision})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFAsset:
        return cls(
            kind=str(payload["kind"]),
            name=str(payload["name"]),
            repository=str(payload["repository"]),
            revision=str(payload["revision"]),
            config=None if payload.get("config") is None else str(payload["config"]),
            split=None if payload.get("split") is None else str(payload["split"]),
        )


WorkerRunner = Callable[[str, Sequence[HFAsset]], tuple[HFAsset, ...]]


def resolve_mode(env: Mapping[str, str]) -> str:
    explicit = env.get("RC_PAG_HF_MODE", "").strip().lower()
    if explicit:
        if explicit not in VALID_MODES:
            raise ValueError("RC_PAG_HF_MODE must be auto, offline, or online")
        return explicit
    legacy = env.get("RC_PAG_HF_OFFLINE", "").strip().lower()
    if legacy in {"1", "true", "yes"}:
        return "offline"
    if legacy in {"0", "false", "no"}:
        return "online"
    if legacy:
        raise ValueError("RC_PAG_HF_OFFLINE must be a boolean value")
    return "auto"


def discover_assets(config: RCPAGConfig) -> tuple[HFAsset, ...]:
    assets: list[HFAsset] = []
    for name, spec in sorted(config.models.items()):
        assets.append(
            HFAsset(
                kind="model",
                name=name,
                repository=spec.repository,
                revision=spec.revision,
            )
        )
    for pool, spec in sorted(config.datasets.items()):
        for dataset_config in sorted(spec.configs):
            assets.append(
                HFAsset(
                    kind="dataset",
                    name=f"{pool}:{dataset_config}:{spec.split}",
                    repository=spec.path,
                    revision=spec.revision,
                    config=dataset_config,
                    split=spec.split,
                )
            )

    unique: dict[tuple[str, str, str, str | None, str | None], HFAsset] = {}
    for asset in assets:
        key = (
            asset.kind,
            asset.repository,
            asset.revision,
            asset.config,
            asset.split,
        )
        unique.setdefault(key, asset)
    return tuple(unique.values())


def worker_environment(*, offline: bool) -> dict[str, str]:
    environment = os.environ.copy()
    value = "1" if offline else "0"
    for name in OFFLINE_VARIABLES:
        environment[name] = value
    environment.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    environment.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    return environment


def _format_missing(assets: Sequence[HFAsset]) -> str:
    return ", ".join(asset.identity for asset in assets)


def bootstrap_assets(
    assets: Sequence[HFAsset],
    *,
    mode: str,
    worker_runner: Callable[..., tuple[HFAsset, ...]] | None = None,
) -> None:
    if mode not in VALID_MODES:
        raise ValueError("mode must be auto, offline, or online")
    run_worker = worker_runner or isolated_worker
    all_assets = tuple(assets)

    if mode == "online":
        failed = run_worker("fetch", all_assets, offline=False)
        if failed:
            raise BootstrapError("online fetch failed for " + _format_missing(failed))
        print(f"HF bootstrap: resolved {len(all_assets)} pinned assets online.", flush=True)
        return

    missing = run_worker("probe", all_assets, offline=True)
    if not missing:
        print(f"HF bootstrap: all {len(all_assets)} pinned assets are cached.", flush=True)
        return
    if mode == "offline":
        raise BootstrapError(
            "strict offline mode is missing pinned assets: " + _format_missing(missing)
        )

    print(
        f"HF bootstrap: {len(missing)} asset(s) missing; temporarily enabling online fetch.",
        flush=True,
    )
    failed = run_worker("fetch", missing, offline=False)
    if failed:
        raise BootstrapError(
            "could not download pinned assets: "
            + _format_missing(failed)
            + ". Pre-populate the shared cache on a network-enabled node and resubmit."
        )
    remaining = run_worker("probe", all_assets, offline=True)
    if remaining:
        raise BootstrapError(
            "download completed but strict offline verification still failed for: "
            + _format_missing(remaining)
        )
    print(
        f"HF bootstrap: downloaded {len(missing)} missing asset(s); offline verification passed.",
        flush=True,
    )


def _safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "<redacted>")
    return message[-2000:]


def _materialize(asset: HFAsset, *, offline: bool) -> None:
    if asset.kind == "model":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=asset.repository,
            revision=asset.revision,
            local_files_only=offline,
        )
        return
    if asset.kind == "dataset":
        from datasets import load_dataset

        load_dataset(
            asset.repository,
            asset.config,
            split=asset.split,
            revision=asset.revision,
        )
        return
    raise ValueError(f"unsupported Hugging Face asset kind: {asset.kind}")


def _read_assets(path: Path) -> tuple[HFAsset, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(HFAsset.from_dict(item) for item in payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker(action: str, assets_path: Path, result_path: Path) -> int:
    if action not in {"probe", "fetch"}:
        raise ValueError(f"unknown worker action: {action}")
    assets = _read_assets(assets_path)
    offline = action == "probe"
    attempts = 1 if offline else max(1, int(os.environ.get("RC_PAG_HF_FETCH_ATTEMPTS", "3")))
    failures = []
    for asset in assets:
        last_error = "unknown failure"
        for attempt in range(1, attempts + 1):
            try:
                _materialize(asset, offline=offline)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - persisted as an asset-level diagnostic
                last_error = _safe_error(exc)
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if last_error:
            failures.append({"asset": asset.to_dict(), "error": last_error})
    _write_json(result_path, {"action": action, "failures": failures})
    return 0


def isolated_worker(
    action: str,
    assets: Sequence[HFAsset],
    *,
    offline: bool,
) -> tuple[HFAsset, ...]:
    with tempfile.TemporaryDirectory(prefix="rc-pag-hf-") as temporary:
        temporary_path = Path(temporary)
        assets_path = temporary_path / "assets.json"
        result_path = temporary_path / "result.json"
        assets_path.write_text(
            json.dumps([asset.to_dict() for asset in assets], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            action,
            "--assets-file",
            str(assets_path),
            "--result-file",
            str(result_path),
        )
        completed = subprocess.run(
            command,
            check=False,
            env=worker_environment(offline=offline),
        )
        if completed.returncode != 0 or not result_path.is_file():
            raise BootstrapError(
                f"Hugging Face {action} worker failed with exit code {completed.returncode}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        failures = payload.get("failures", ())
        for failure in failures:
            failed_asset = HFAsset.from_dict(failure["asset"])
            print(
                f"HF bootstrap {action} failed for {failed_asset.identity}: {failure['error']}",
                file=sys.stderr,
                flush=True,
            )
        return tuple(HFAsset.from_dict(failure["asset"]) for failure in failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate and verify pinned Hugging Face assets for an RC-PAG run."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=sorted(VALID_MODES))
    parser.add_argument("--worker", choices=("probe", "fetch"), help=argparse.SUPPRESS)
    parser.add_argument("--assets-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        if args.assets_file is None or args.result_file is None:
            parser.error("--worker requires --assets-file and --result-file")
        return _worker(args.worker, args.assets_file, args.result_file)
    if args.config is None:
        parser.error("--config is required")
    mode = args.mode or resolve_mode(os.environ)
    assets = discover_assets(load_rc_pag_config(args.config))
    print(f"HF bootstrap mode={mode}; checking {len(assets)} pinned assets.", flush=True)
    try:
        bootstrap_assets(assets, mode=mode)
    except BootstrapError as exc:
        print(f"HF bootstrap failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
