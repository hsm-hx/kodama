"""Export/import keeps meaning; bad files are rejected without leaving a usable target."""

from __future__ import annotations

import json

import pytest

from kodama import migration
from kodama.domain import MemoryStatus, PersonaStatus
from kodama.storage.base import ImportConflict, ImportValidationError
from kodama.storage.sqlite import SQLiteStore
from seed import REN_V1, REN_V2, seed_store


@pytest.fixture
def exported(tmp_path):
    src_path = tmp_path / "src.db"
    src = SQLiteStore(src_path)
    ids = seed_store(src)
    export = tmp_path / "export.json"
    migration.export_to_file(src, export, settings={"api_key": "sk-ant-secret", "effort": "low"})
    yield src, ids, export, src_path
    src.close()


def _semantic(store) -> dict:
    snap = store.export_snapshot()
    return {k: sorted(snap[k], key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)) for k in snap}


def _rewrite(path, mutate):
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    doc["counts"] = {s: len(doc["data"][s]) for s in doc["data"]}
    doc["checksum"] = migration.checksum(doc["data"])
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def test_roundtrip_preserves_meaning(exported, tmp_path):
    src, ids, export, src_path = exported
    target = tmp_path / "new.db"
    result = migration.import_file(export, target, active_db_path=src_path)
    assert result.created_new
    dst = SQLiteStore(target)
    try:
        before, after = _semantic(src), _semantic(dst)
        settings_before = before.pop("settings")
        settings_after = after.pop("settings")
        assert before == after  # ids, counts, order fields, sources, links, states, histories
        assert {r["key"]: r["value"] for r in settings_after} == {
            **{r["key"]: r["value"] for r in settings_before},
            "effort": "low",
        }
        assert [(m.speaker.value, m.text) for m in dst.list_messages(ids.session1)] == list(ids.transcript)
        assert dst.get_active_persona("ren").body == REN_V2
        assert dst.get_persona_version(ids.persona_ren_v1).body == REN_V1
        assert dst.get_persona_version(ids.persona_ren_v1).status == PersonaStatus.RETIRED
        assert dst.get_memory_version(ids.mem_room_old).status == MemoryStatus.SUPERSEDED
        assert dst.get_memory_version(ids.mem_room_new).supersedes_version_id == ids.mem_room_old
        assert dst.get_source_excerpt(ids.excerpt).text == src.get_source_excerpt(ids.excerpt).text
    finally:
        dst.close()


def test_export_contains_no_secrets(exported):
    _, _, export, _ = exported
    text = export.read_text(encoding="utf-8")
    assert "sk-ant" not in text
    assert "api_key" not in text
    doc = json.loads(text)
    assert doc["schema_version"] == 1 and doc["format"] == "kodama-export"
    assert migration.verify_file(export).ok


def test_reimport_is_idempotent(exported, tmp_path):
    _, _, export, src_path = exported
    target = tmp_path / "new.db"
    migration.import_file(export, target, active_db_path=src_path)
    store = SQLiteStore(target)
    before = store.export_snapshot()
    store.close()
    again = migration.import_file(export, target, active_db_path=src_path)
    assert not again.created_new
    assert sum(again.inserted.values()) == 0
    store = SQLiteStore(target)
    assert store.export_snapshot() == before
    store.close()


def test_same_id_different_content_conflicts(exported, tmp_path):
    _, ids, export, src_path = exported
    target = tmp_path / "new.db"
    migration.import_file(export, target, active_db_path=src_path)
    store = SQLiteStore(target)
    before = store.export_snapshot()
    store.close()

    changed = tmp_path / "changed.json"
    changed.write_text(export.read_text(encoding="utf-8"), encoding="utf-8")

    def mutate(doc):
        for m in doc["data"]["messages"]:
            if m["id"] == ids.user_message_ids[0]:
                m["text"] = "書き換えられた原文"

    _rewrite(changed, mutate)
    with pytest.raises(ImportConflict):
        migration.import_file(changed, target, active_db_path=src_path)
    store = SQLiteStore(target)
    assert store.export_snapshot() == before
    store.close()


def test_refuses_active_database(exported):
    _, _, export, src_path = exported
    with pytest.raises(ImportValidationError):
        migration.import_file(export, src_path, active_db_path=src_path)


def _assert_rejected(path, tmp_path, src_path):
    assert not migration.verify_file(path).ok
    target = tmp_path / "rejected.db"
    with pytest.raises(ImportValidationError):
        migration.import_file(path, target, active_db_path=src_path)
    assert not target.exists()
    assert [p.name for p in tmp_path.iterdir() if "importing" in p.name] == []


def test_rejects_broken_reference(exported, tmp_path):
    _, _, export, src_path = exported

    def mutate(doc):
        doc["data"]["memory_versions"][0]["source_message_ids"] = ["00000000-0000-4000-8000-000000000000"]

    _rewrite(export, mutate)
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_dangling_link(exported, tmp_path):
    _, _, export, src_path = exported

    def mutate(doc):
        doc["data"]["links"][0]["dst_id"] = "00000000-0000-4000-8000-000000000000"

    _rewrite(export, mutate)
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_superseded_without_successor(exported, tmp_path):
    _, ids, export, src_path = exported

    def mutate(doc):
        doc["data"]["memory_versions"] = [
            v for v in doc["data"]["memory_versions"] if v["id"] != ids.mem_room_new
        ]
        doc["data"]["links"] = [
            l for l in doc["data"]["links"] if ids.mem_room_new not in (l["src_id"], l["dst_id"])
        ]

    _rewrite(export, mutate)
    report = migration.verify_file(export)
    assert any("superseded without a successor" in e for e in report.errors)
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_unsupported_schema_version(exported, tmp_path):
    _, _, export, src_path = exported
    _rewrite(export, lambda doc: doc.update(schema_version=99))
    report = migration.verify_file(export)
    assert any("schema_version" in e for e in report.errors)
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_truncated_file(exported, tmp_path):
    _, _, export, src_path = exported
    raw = export.read_bytes()
    export.write_bytes(raw[: len(raw) // 2])
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_checksum_mismatch(exported, tmp_path):
    _, _, export, src_path = exported
    doc = json.loads(export.read_text(encoding="utf-8"))
    doc["data"]["messages"][0]["text"] = "黙って改変"
    export.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    report = migration.verify_file(export)
    assert any("checksum" in e for e in report.errors)
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_naive_timestamp_and_bad_enum(exported, tmp_path):
    _, _, export, src_path = exported

    def mutate(doc):
        doc["data"]["messages"][0]["created_at"] = "2026-10-08T10:00:00"
        doc["data"]["messages"][1]["speaker"] = "narrator"

    _rewrite(export, mutate)
    report = migration.verify_file(export)
    assert len(report.errors) >= 2
    _assert_rejected(export, tmp_path, src_path)


def test_rejects_secret_in_settings(exported, tmp_path):
    _, _, export, src_path = exported
    _rewrite(export, lambda doc: doc["data"]["settings"].append({"key": "api_key", "value": "sk-ant-x"}))
    _assert_rejected(export, tmp_path, src_path)


def test_verify_does_not_touch_files(exported, tmp_path):
    _, _, export, _ = exported
    before = sorted(p.name for p in tmp_path.iterdir())
    assert migration.verify_file(export).ok
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def _strip_cache_keys(doc):
    for t in doc["data"]["turns"]:
        t.pop("cache_read_tokens", None)
        t.pop("cache_write_tokens", None)


def test_export_contains_cache_usage(exported):
    doc = json.loads(exported[2].read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert any(t["cache_read_tokens"] == 900 and t["cache_write_tokens"] == 30 for t in doc["data"]["turns"])
    assert migration.verify_file(exported[2]).ok


def test_old_export_without_cache_keys_is_accepted(exported, tmp_path):
    _, _, export, src_path = exported
    old = tmp_path / "old.json"
    old.write_text(export.read_text(encoding="utf-8"), encoding="utf-8")
    _rewrite(old, _strip_cache_keys)
    assert migration.verify_file(old).ok
    target = tmp_path / "new.db"
    migration.import_file(old, target, active_db_path=src_path)
    store = SQLiteStore(target)
    turns = [t for s in store.list_sessions() for t in store.list_turns(s.id)]
    assert turns and all(t.cache_read_tokens is None and t.cache_write_tokens is None for t in turns)
    store.close()


def test_old_export_reimport_over_new_data_is_not_a_conflict(exported, tmp_path):
    # 新しい export を取り込んだ後でも、同内容の旧 export（キー欠落）を再 import しても衝突しない。
    # キー欠落と None は同一視される。
    _, _, export, src_path = exported
    old = tmp_path / "old.json"
    old.write_text(export.read_text(encoding="utf-8"), encoding="utf-8")

    def mutate(doc):
        for t in doc["data"]["turns"]:
            if t["cache_read_tokens"] is None and t["cache_write_tokens"] is None:
                t.pop("cache_read_tokens"); t.pop("cache_write_tokens")

    _rewrite(old, mutate)
    target = tmp_path / "new.db"
    migration.import_file(export, target, active_db_path=src_path)
    again = migration.import_file(old, target, active_db_path=src_path)
    assert sum(again.inserted.values()) == 0
    # 値ありのターンが違えば衝突する
    changed = tmp_path / "changed.json"
    changed.write_text(export.read_text(encoding="utf-8"), encoding="utf-8")
    _rewrite(changed, lambda d: [t.update(cache_read_tokens=1) for t in d["data"]["turns"] if t["cache_read_tokens"] == 900])
    with pytest.raises(ImportConflict):
        migration.import_file(changed, target, active_db_path=src_path)
