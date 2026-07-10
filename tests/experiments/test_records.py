from __future__ import annotations

from pag.experiments.records import RecordStore


def test_record_round_trip_and_resume(tmp_path) -> None:
    store = RecordStore(tmp_path, identity={"config_hash": "abc"})
    store.write("development", "pag", "sample-1", {"total_nfe": 9})
    assert store.is_complete("development", "pag", "sample-1")
    assert store.read("development", "pag", "sample-1")["total_nfe"] == 9


def test_mismatched_record_is_quarantined(tmp_path) -> None:
    first = RecordStore(tmp_path, identity={"config_hash": "old"})
    first.write("development", "pag", "sample-1", {"total_nfe": 9})
    second = RecordStore(tmp_path, identity={"config_hash": "new"})
    assert not second.is_complete("development", "pag", "sample-1")
    assert list((tmp_path / "quarantine").glob("*.json"))
