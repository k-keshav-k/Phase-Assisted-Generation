from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, sort_keys=True, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)


def _record_name(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_")[:80] or "sample"
    digest = hashlib.sha256(sample_id.encode()).hexdigest()[:10]
    return f"{readable}-{digest}.json"


@dataclass(slots=True)
class RecordStore:
    root: Path
    identity: dict[str, Any]

    def __init__(self, root: str | Path, identity: dict[str, Any]) -> None:
        self.root = Path(root)
        self.identity = dict(identity)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, stage: str, method: str, sample_id: str) -> Path:
        return self.root / stage / method / _record_name(sample_id)

    def write(
        self,
        stage: str,
        method: str,
        sample_id: str,
        payload: dict[str, Any],
    ) -> Path:
        record = {
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "stage": stage,
            "method": method,
            "sample_id": sample_id,
            "created_at": datetime.now(UTC).isoformat(),
            **payload,
        }
        target = self.path_for(stage, method, sample_id)
        _atomic_json(target, record)
        return target

    def _load_valid(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._quarantine(path, "invalid_json")
            return None
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("identity") != self.identity
        ):
            self._quarantine(path, "identity_mismatch")
            return None
        return payload

    def _quarantine(self, path: Path, reason: str) -> None:
        if not path.exists():
            return
        quarantine = self.root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        target = quarantine / f"{reason}-{timestamp}-{path.name}"
        os.replace(path, target)

    def is_complete(self, stage: str, method: str, sample_id: str) -> bool:
        path = self.path_for(stage, method, sample_id)
        return path.exists() and self._load_valid(path) is not None

    def read(self, stage: str, method: str, sample_id: str) -> dict[str, Any]:
        payload = self._load_valid(self.path_for(stage, method, sample_id))
        if payload is None:
            raise FileNotFoundError(f"no valid record for {stage}/{method}/{sample_id}")
        return payload

    def records(self, stage: str, method: str) -> list[dict[str, Any]]:
        directory = self.root / stage / method
        if not directory.exists():
            return []
        values = [
            value for path in sorted(directory.glob("*.json")) if (value := self._load_valid(path))
        ]
        return sorted(values, key=lambda item: str(item["sample_id"]))

    def paired_records(self, stage: str, methods: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        result = {method: self.records(stage, method) for method in methods}
        id_sets = {method: {row["sample_id"] for row in rows} for method, rows in result.items()}
        if len({frozenset(ids) for ids in id_sets.values()}) > 1:
            raise ValueError(f"incomplete paired coverage: {id_sets}")
        return result

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        target = self.root / "manifest.json"
        _atomic_json(target, {"schema_version": SCHEMA_VERSION, **payload})
        return target

    def write_named(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.root / name
        _atomic_json(target, payload)
        return target
