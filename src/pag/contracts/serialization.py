from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


def from_dict(data_type: type[T], payload: Any) -> T:
    return _restore(data_type, payload)


def dump_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_json(path: str | Path, data_type: type[T]) -> T:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(data_type, payload)


def dump_jsonl(path: str | Path, items: list[Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(to_dict(item), sort_keys=True) for item in items]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def load_jsonl(path: str | Path, item_type: type[T]) -> list[T]:
    records: list[T] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(from_dict(item_type, json.loads(line)))
    return records


def _restore(annotation: Any, payload: Any) -> Any:
    if payload is None:
        return None
    if annotation is Any:
        return payload
    origin = get_origin(annotation)
    if origin in (list, tuple):
        (item_type,) = get_args(annotation)
        return [_restore(item_type, item) for item in payload]
    if origin is dict:
        key_type, value_type = get_args(annotation)
        if key_type not in (Any, str):
            return payload
        return {key: _restore(value_type, value) for key, value in payload.items()}
    if origin in (Union, getattr(__import__("types"), "UnionType", Union)):
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none_args) == 1:
            return _restore(non_none_args[0], payload)
        return payload
    if isinstance(annotation, type) and is_dataclass(annotation):
        type_hints = get_type_hints(annotation)
        kwargs = {}
        for field in fields(annotation):
            field_type = type_hints.get(field.name, field.type)
            if field.name in payload:
                kwargs[field.name] = _restore(field_type, payload[field.name])
        return annotation(**kwargs)
    return payload
