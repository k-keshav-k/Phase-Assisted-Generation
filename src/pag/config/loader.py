from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pag.contracts.schemas import RunConfig, SampleRecord
from pag.contracts.serialization import from_dict


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    payload = _expand_refs(load_yaml(config_path), config_path.parent)
    payload["dataset_path"] = str(_resolve_path(payload["dataset_path"], config_path.parent))
    return from_dict(RunConfig, payload)


def load_samples(path: str | Path) -> list[SampleRecord]:
    payload = load_yaml(path)
    raw_samples = (
        payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    )
    return [from_dict(SampleRecord, item) for item in raw_samples]


def _expand_refs(node: Any, base_dir: Path) -> Any:
    if isinstance(node, dict):
        if "from_file" in node:
            ref_path = _resolve_path(node["from_file"], base_dir)
            loaded = _expand_refs(load_yaml(ref_path), ref_path.parent)
            overrides = {key: value for key, value in node.items() if key != "from_file"}
            if overrides:
                if not isinstance(loaded, dict):
                    msg = f"Cannot apply overrides to non-mapping config reference: {ref_path}"
                    raise ValueError(msg)
                loaded = {**loaded, **_expand_refs(overrides, base_dir)}
            return loaded
        return {key: _expand_refs(value, base_dir) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_refs(item, base_dir) for item in node]
    return node


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()
