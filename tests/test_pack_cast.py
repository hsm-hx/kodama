"""人物設定パックと cast（参加者）: DB・移行ファイル・プロンプト・CLI での扱い。"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from kodama import cli, migration
from kodama.config import Config, ConfigError, load_config
from kodama.context import build_context, output_rules
from kodama.conversation import sync_personas
from kodama.domain import Cast, CastMember
from kodama.model.mock import MockAdapter
from kodama.personas import load_pack, load_persona_files
from kodama.reply import InvalidReply, parse_reply, reply_schema
from kodama.storage.base import ImportValidationError, InvalidState
from kodama.storage.sqlite import SQLiteStore
from kodama.voice_check import load_cases
from seed import CAST, EXAMPLE_PACK_DIR, PACK, seed_store

OTHER_CAST = Cast(
    user=CastMember("me", "わたし"),
    characters=(CastMember("kai", "カイ"), CastMember("mio", "ミオ")),
)


def _make_legacy_db(path: Path) -> None:
    """旧スキーマ（v2: 話者IDを固定の CHECK 制約で縛り、cast を記録していない）のDBを作る。"""
    store = SQLiteStore(path, cast=CAST)
    seed_store(store)
    store.close()
    ids = ", ".join(f"'{i}'" for i in CAST.speaker_ids)
    chars = ", ".join(f"'{i}'" for i in CAST.character_ids)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = OFF")
    for table, column, allowed in (("messages", "speaker", ids), ("memory_versions", "perspective", chars)):
        sql = con.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
        if column == "speaker":
            legacy = sql.replace("speaker TEXT NOT NULL,", f"speaker TEXT NOT NULL CHECK (speaker IN ({allowed})),")
        else:
            legacy = sql.replace(
                "perspective TEXT,", f"perspective TEXT CHECK (perspective IS NULL OR perspective IN ({allowed})),"
            )
        assert legacy != sql
        con.execute(legacy.replace(f"CREATE TABLE {table}", f"CREATE TABLE {table}__old", 1))
        con.execute(f"INSERT INTO {table}__old SELECT * FROM {table}")
        con.execute(f"DROP TABLE {table}")
        con.execute(f"ALTER TABLE {table}__old RENAME TO {table}")
    con.execute("DELETE FROM schema_info WHERE key = 'cast'")
    con.execute("UPDATE schema_info SET value = '2' WHERE key = 'schema_version'")
    con.commit()
    con.close()


def _snapshot_without_ids(store) -> dict:
    snap = store.export_snapshot()
    return {k: sorted(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in v) for k, v in snap.items()}


# ---------------------------------------------------------------- DB と cast


def test_new_db_records_cast_and_rejects_other_pack(tmp_path):
    path = tmp_path / "k.db"
    SQLiteStore(path, cast=CAST).close()
    reopened = SQLiteStore(path)  # cast を渡さなくても記録から読める
    assert reopened.get_cast() == CAST
    reopened.close()
    with pytest.raises(InvalidState):
        SQLiteStore(path, cast=OTHER_CAST)


def test_display_name_change_is_allowed(tmp_path):
    path = tmp_path / "k.db"
    SQLiteStore(path, cast=CAST).close()
    renamed = Cast(user=CastMember(CAST.user_id, "きみ"), characters=CAST.characters)
    s = SQLiteStore(path, cast=renamed)
    s.close()
    assert SQLiteStore(path).get_cast().user.display_name == "きみ"


def test_legacy_db_is_upgraded_in_place_without_losing_data(tmp_path):
    path = tmp_path / "legacy.db"
    _make_legacy_db(path)
    con = sqlite3.connect(path)
    before = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("messages", "memory_versions", "turns")}
    con.close()

    store = SQLiteStore(path, cast=CAST)
    try:
        assert store.get_cast() == CAST
        con = sqlite3.connect(path)
        ddl = " ".join(r[0] for r in con.execute("SELECT sql FROM sqlite_master WHERE name IN ('messages','memory_versions')"))
        version = con.execute("SELECT value FROM schema_info WHERE key='schema_version'").fetchone()[0]
        after = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in before}
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        con.close()
        assert "CHECK (speaker" not in ddl and "CHECK (perspective" not in ddl
        assert version == "3" and after == before and fk == []
        assert store.list_memory_versions()  # まだ読める
    finally:
        store.close()


def test_legacy_db_with_speakers_outside_pack_is_rejected(tmp_path):
    path = tmp_path / "legacy.db"
    _make_legacy_db(path)
    with pytest.raises(InvalidState):
        SQLiteStore(path, cast=OTHER_CAST)


def test_store_rejects_speakers_and_persona_keys_outside_cast(tmp_path):
    store = SQLiteStore(tmp_path / "k.db", cast=CAST)
    try:
        s = store.create_session("t")
        turn, msg = store.begin_turn(s.id, "やあ", "mock", "mock")
        assert msg.speaker == CAST.user_id
        with pytest.raises(ValueError):
            store.complete_turn(turn.id, [("narrator", "地の文")])
        with pytest.raises(ValueError):
            store.complete_turn(turn.id, [(CAST.user_id, "なりすまし")])
        with pytest.raises(ValueError):
            store.activate_persona_version("stranger", "本文")
    finally:
        store.close()


# ---------------------------------------------------------------- 移行ファイル


@pytest.fixture
def export_v2(tmp_path):
    src_path = tmp_path / "src.db"
    src = SQLiteStore(src_path, cast=CAST)
    seed_store(src)
    path = tmp_path / "e.json"
    migration.export_to_file(src, path)
    yield src, src_path, path
    src.close()


def _as_v1(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["schema_version"] = 1
    del doc["data"]["cast"]
    doc["checksum"] = migration.checksum(doc["data"])
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def test_export_v2_carries_cast(export_v2):
    _, _, path = export_v2
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 2
    assert Cast.from_dict(doc["data"]["cast"]) == CAST
    assert migration.verify_file(path).cast == CAST


def test_v2_file_is_rejected_by_a_pack_with_other_ids(export_v2, tmp_path):
    _, src_path, path = export_v2
    assert not migration.verify_file(path, cast=OTHER_CAST).ok
    with pytest.raises(ImportValidationError):
        migration.import_file(path, tmp_path / "x.db", active_db_path=src_path, cast=OTHER_CAST)
    assert not (tmp_path / "x.db").exists()


def test_v1_file_uses_target_pack_cast(export_v2, tmp_path):
    src, src_path, path = export_v2
    _as_v1(path)
    with pytest.raises(ImportValidationError):  # cast がなければ取り込み先を決められない
        migration.import_file(path, tmp_path / "nocast.db", active_db_path=src_path)
    assert not (tmp_path / "nocast.db").exists()
    assert not migration.verify_file(path, cast=OTHER_CAST).ok  # 話者がパックに含まれない
    result = migration.import_file(path, tmp_path / "v1.db", active_db_path=src_path, cast=CAST)
    assert result.created_new
    dst = SQLiteStore(tmp_path / "v1.db")
    try:
        assert dst.get_cast() == CAST
        assert _snapshot_without_ids(dst) == _snapshot_without_ids(src)
    finally:
        dst.close()


def test_existing_target_with_other_cast_is_rejected(export_v2, tmp_path):
    _, src_path, path = export_v2
    other = tmp_path / "other.db"
    SQLiteStore(other, cast=OTHER_CAST).close()
    with pytest.raises(ImportValidationError):
        migration.import_file(path, other, active_db_path=src_path)


# ---------------------------------------------------------------- プロンプト・返答


def test_output_rules_and_schema_come_from_cast():
    rules = output_rules(OTHER_CAST)
    assert "カイ" in rules and "ミオ" in rules and "わたし" in rules
    assert '"kai" または "mio"' in rules
    for name in CAST.character_ids + tuple(m.display_name for m in (CAST.user, *CAST.characters)):
        assert name not in rules
    schema = reply_schema(OTHER_CAST.character_ids)
    assert schema["properties"]["utterances"]["items"]["properties"]["speaker"]["enum"] == ["kai", "mio"]
    with pytest.raises(InvalidReply):
        parse_reply('{"utterances": [{"speaker": "ren", "text": "x"}], "memory_candidates": []}', ["kai", "mio"])


def test_context_uses_pack_names(tmp_path):
    cfg = Config(db_path=str(tmp_path / "k.db"), persona_pack=str(EXAMPLE_PACK_DIR))
    store = SQLiteStore(cfg.db_path, cast=CAST)
    try:
        sync_personas(store, PACK)
        s = store.create_session("t")
        plan = build_context(store, cfg, PACK, s.id, "蓮、ただいま")
        assert len(plan.system_blocks) == len(PACK.persona_keys) + 1
        assert "蓮" in plan.system_blocks[-1] and "あなた" in plan.system_blocks[-1]
        from kodama.context import to_model_request

        req = to_model_request(plan, cfg)
        assert req.output_schema == reply_schema(CAST.character_ids)
        res = MockAdapter().generate(req)
        assert [s for s, _ in parse_reply(res.raw_text, CAST.character_ids).utterances] == ["ren"]
    finally:
        store.close()


def test_mock_addressing_by_alias():
    chars = [{"id": c.id, "display_name": c.display_name, "aliases": list(c.address_aliases)} for c in PACK.characters]
    from kodama.model.mock import choose_speakers

    assert choose_speakers("れん、おはよう", chars) == ["ren"]
    assert choose_speakers("あおい、おはよう", chars) == ["aoi"]


# ---------------------------------------------------------------- 設定・CLI


def test_old_config_key_is_explained(tmp_path):
    p = tmp_path / "k.toml"
    p.write_text('personas_dir = "personas"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "persona_pack" in str(e.value)


def test_cli_refuses_db_of_another_pack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "k.db"
    SQLiteStore(db, cast=OTHER_CAST).close()
    out: list[str] = []
    code = cli.main(["--db", str(db), "--pack", str(EXAMPLE_PACK_DIR)], lambda p: "/quit", out.append)
    assert code == 2 and any("DBを開けません" in l for l in out)


def test_cli_bad_pack_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out: list[str] = []
    code = cli.main(["--db", str(tmp_path / "k.db"), "--pack", str(tmp_path / "nope")], lambda p: "/quit", out.append)
    assert code == 2 and any("人物設定パック" in l for l in out)


# ---------------------------------------------------------------- 私的パック（任意）


@pytest.mark.skipif(not os.environ.get("KODAMA_PRIVATE_PACK"), reason="KODAMA_PRIVATE_PACK が未設定")
def test_private_pack_loads_and_builds_prompt(tmp_path):
    pack = load_pack(os.environ["KODAMA_PRIVATE_PACK"])
    files = load_persona_files(pack)
    assert set(files) == set(pack.persona_keys)
    cfg = Config(db_path=str(tmp_path / "p.db"), persona_pack=pack.root)
    store = SQLiteStore(cfg.db_path, cast=pack.cast)
    try:
        sync_personas(store, pack)
        s = store.create_session("t")
        plan = build_context(store, cfg, pack, s.id, "こんにちは")
        assert all(b.strip() for b in plan.system_blocks)
    finally:
        store.close()
    if pack.voice_cases_path():
        data = load_cases(pack.voice_cases_path())
        assert set(data["base_checks"]["allowed_speakers"]) == set(pack.cast.character_ids)
        for c in data["cases"]:
            assert c["id"] and c["input"] and c["human_review"]
