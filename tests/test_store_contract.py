"""Store contract: the same logical assertions must hold for a fresh SQLite store and for one
rebuilt from an export file."""

from __future__ import annotations

import pytest

from kodama.domain import (
    MemoryDraft,
    MemoryKind,
    MemoryOrigin,
    MemoryStatus,
    NodeType,
    PersonaStatus,
    TurnStatus,
    parse_iso,
)
from kodama.storage.base import InvalidState, NotFound
from seed import CAST, COMMON_V1, REN_V1, REN_V2, SETTINGS, AOI_V1


def test_sessions_and_transcript_order(store, seed):
    titles = [s.title for s in store.list_sessions()]
    assert titles == ["初日", "翌日"]
    msgs = store.list_messages(seed.session1)
    assert [(m.speaker, m.text) for m in msgs] == list(seed.transcript)
    assert [m.seq for m in msgs] == list(range(len(msgs)))
    assert store.list_messages(seed.session2) == []


def test_timestamps_keep_timezone(store, seed):
    for m in store.list_messages(seed.session1):
        assert parse_iso(m.created_at).utcoffset() is not None
        assert m.created_at.endswith("+09:00")


def test_turn_statuses_and_usage(store, seed):
    turns = store.list_turns(seed.session1)
    assert [t.id for t in turns] == list(seed.turn_ids)
    assert tuple(t.status for t in turns) == seed.turn_statuses
    assert (turns[0].usage_input_tokens, turns[0].usage_output_tokens) == (120, 40)
    assert turns[1].usage_input_tokens is None  # unknown stays unknown
    assert (turns[0].cache_read_tokens, turns[0].cache_write_tokens) == (900, 30)
    assert (turns[1].cache_read_tokens, turns[1].cache_write_tokens) == (None, None)
    assert turns[2].error == "timeout"
    assert (turns[2].cache_read_tokens, turns[2].cache_write_tokens) == (7, 0)
    assert seed.persona_ren_v2 in turns[0].persona_version_ids


def test_recent_messages_mark_incomplete_turns(store, seed):
    recent = store.recent_messages(seed.session1, 3)
    assert [e.message.text for e in recent] == [t for _, t in seed.transcript[-3:]]
    assert [e.turn_status for e in recent] == [
        TurnStatus.COMPLETED,
        TurnStatus.FAILED,
        TurnStatus.INTERRUPTED,
    ]


def test_persona_versions_keep_bodies(store, seed):
    active = store.get_active_persona("ren")
    assert active.id == seed.persona_ren_v2 and active.body == REN_V2
    old = store.get_persona_version(seed.persona_ren_v1)
    assert old.status == PersonaStatus.RETIRED and old.body == REN_V1
    assert store.get_active_persona("aoi").body == AOI_V1
    assert store.get_active_persona("common").body == COMMON_V1
    assert [v.id for v in store.list_persona_versions("ren")] == [seed.persona_ren_v1, seed.persona_ren_v2]


def test_persona_activation_is_idempotent_and_versioned(store, seed):
    same = store.activate_persona_version("ren", REN_V2)
    assert same.id == seed.persona_ren_v2
    v3 = store.activate_persona_version("ren", REN_V2 + "追記\n", note="追記")
    assert store.get_active_persona("ren").id == v3.id
    assert store.get_persona_version(seed.persona_ren_v2).status == PersonaStatus.RETIRED
    assert len(store.list_persona_versions("ren")) == 3


def test_memory_states(store, seed):
    get = store.get_memory_version
    assert get(seed.mem_coffee).status == MemoryStatus.APPROVED
    assert get(seed.mem_clothes_view).status == MemoryStatus.APPROVED
    assert get(seed.mem_clothes_view).perspective == "aoi"
    assert get(seed.mem_clothes_view).origin == MemoryOrigin.MODEL_CANDIDATE
    assert get(seed.mem_imagination_rejected).status == MemoryStatus.REJECTED
    assert get(seed.mem_imagination_rejected).status_reason == "想像は残さない"
    assert get(seed.mem_pending_candidate).status == MemoryStatus.CANDIDATE
    assert get(seed.mem_invalidated).status == MemoryStatus.INVALIDATED
    # candidates produced by a turn point at that turn's user message
    assert get(seed.mem_pending_candidate).source_message_ids == (seed.user_message_ids[1],)
    assert get(seed.mem_coffee).occurred_at is None
    assert get(seed.mem_milk).occurred_at == "2026-10-07T08:00:00+09:00"


def test_revision_history(store, seed):
    old = store.get_memory_version(seed.mem_room_old)
    new = store.get_memory_version(seed.mem_room_new)
    assert old.status == MemoryStatus.SUPERSEDED and old.status_reason == "本人の訂正"
    assert new.status == MemoryStatus.APPROVED and new.supersedes_version_id == old.id
    assert new.memory_id == old.memory_id
    assert new.source_message_ids == old.source_message_ids  # inherited source
    versions = store.list_memory_versions(memory_id=old.memory_id)
    assert {v.id for v in versions} == {old.id, new.id}
    links = store.links_of(NodeType.MEMORY_VERSION, new.id)
    assert any(l.relation == "supersedes" and l.dst_id == old.id for l in links)


def test_revise_again_and_invalid_transitions(store, seed):
    newer = store.revise_memory(
        seed.mem_room_new,
        MemoryDraft(body="作業部屋は一階の奥", kind=MemoryKind.USER_STATED),
        reason="詳細化",
    )
    approved = store.list_memory_versions([MemoryStatus.APPROVED], memory_id=newer.memory_id)
    assert [v.id for v in approved] == [newer.id]
    with pytest.raises(InvalidState):
        store.revise_memory(seed.mem_room_old, MemoryDraft(body="x", kind=MemoryKind.USER_STATED))
    with pytest.raises(InvalidState):
        store.set_memory_status(seed.mem_imagination_rejected, MemoryStatus.APPROVED)
    with pytest.raises(InvalidState):
        store.set_memory_status(seed.mem_room_old, MemoryStatus.APPROVED)


def test_memory_validation(store, seed):
    with pytest.raises(ValueError):
        store.add_memory(MemoryDraft(body="視点なし", kind=MemoryKind.CHARACTER_VIEW), MemoryOrigin.USER_EXPLICIT)
    with pytest.raises(ValueError):
        store.add_memory(
            MemoryDraft(body="本人発言に視点", kind=MemoryKind.USER_STATED, perspective="ren"),
            MemoryOrigin.USER_EXPLICIT,
        )
    with pytest.raises(NotFound):
        store.add_memory(
            MemoryDraft(body="出典が存在しない", kind=MemoryKind.USER_STATED, source_message_ids=("nope",)),
            MemoryOrigin.USER_EXPLICIT,
        )


def test_search_returns_matches_regardless_of_status(store, seed):
    found = {v.id for v in store.search_memory_versions(["作業部屋"])}
    assert {seed.mem_room_old, seed.mem_room_new} <= found
    by_alias = {v.id for v in store.search_memory_versions(["珈琲"])}
    assert seed.mem_coffee in by_alias
    assert len(store.search_memory_versions(["あなた"], limit=2)) == 2
    assert store.search_memory_versions([]) == []
    assert store.search_memory_versions(["%"]) == []


def test_entities_and_links(store, seed):
    found = store.find_entities(["今朝の珈琲"])
    assert [e.id for e in found] == [seed.entity_coffee]
    coffee_mem = store.get_memory_version(seed.mem_coffee).memory_id
    links = store.links_of(NodeType.ENTITY, seed.entity_coffee)
    assert [(l.relation, l.dst_type, l.dst_id) for l in links] == [("about", NodeType.MEMORY, coffee_mem)]
    related = store.links_of(NodeType.MEMORY, coffee_mem)
    assert len(related) == 3
    assert len(store.links_of(NodeType.MEMORY, coffee_mem, limit=1)) == 1
    again = store.upsert_entity("topic", "コーヒー", ("coffee",))
    assert again.id == seed.entity_coffee and again.aliases == ("珈琲", "coffee")


def test_source_excerpt_and_settings(store, seed):
    excerpt = store.get_source_excerpt(seed.excerpt)
    assert "あおちゃん" in excerpt.text and excerpt.locator == "呼称の節"
    assert store.get_memory_version(seed.mem_excerpt).source_excerpt_ids == (seed.excerpt,)
    assert store.get_settings() == SETTINGS


def test_conversation_continues_after_seed(store, seed):
    turn, msg = store.begin_turn(seed.session1, "続きの話", "mock", "mock-1")
    assert msg.seq == len(seed.transcript)
    assert turn.seq == len(seed.turn_ids)
    with pytest.raises(InvalidState):
        store.begin_turn(seed.session1, "もう一つ", "mock", "mock-1")
    out = store.complete_turn(turn.id, [("aoi", "はい、続けましょう")])
    assert [m.seq for m in out] == [msg.seq + 1]


@pytest.mark.parametrize(
    "utterances",
    [[("user", "なりすまし")], [("narrator", "地の文")], [("ren", "   ")], []],
)
def test_complete_turn_rejects_invalid_replies(store, seed, utterances):
    turn, _ = store.begin_turn(seed.session2, "こんにちは", "mock", "mock-1")
    with pytest.raises(ValueError):
        store.complete_turn(turn.id, utterances)
    assert store.get_turn(turn.id).status == TurnStatus.PENDING
    assert len(store.list_messages(seed.session2)) == 1


def test_complete_and_fail_only_from_pending(store, seed):
    failed = seed.turn_ids[2]
    with pytest.raises(InvalidState):
        store.complete_turn(failed, [("ren", "遅れて届いた返答")])
    assert store.fail_turn(seed.turn_ids[0], TurnStatus.FAILED, "x").status == TurnStatus.COMPLETED



def test_old_schema_db_is_migrated_in_place(tmp_path):
    import sqlite3

    from kodama.domain import Usage
    from kodama.storage.sqlite import SQLiteStore

    path = tmp_path / "old.db"
    s = SQLiteStore(cast=CAST, path=path)
    sess = s.create_session("旧")
    t, _ = s.begin_turn(sess.id, "こんにちは", "mock", "m")
    s.complete_turn(t.id, [("ren", "うん。")], Usage(11, 3))
    s.close()
    # v1 スキーマを再現: 新列を落として schema_version を 1 に戻す
    raw = sqlite3.connect(path)
    raw.execute("ALTER TABLE turns DROP COLUMN cache_read_tokens")
    raw.execute("ALTER TABLE turns DROP COLUMN cache_write_tokens")
    raw.execute("UPDATE schema_info SET value = '1' WHERE key = 'schema_version'")
    raw.commit()
    cols = {r[1] for r in raw.execute("PRAGMA table_info(turns)")}
    assert "cache_read_tokens" not in cols
    raw.close()

    s2 = SQLiteStore(cast=CAST, path=path)
    turns = s2.list_turns(sess.id)
    assert len(turns) == 1 and (turns[0].usage_input_tokens, turns[0].cache_read_tokens) == (11, None)
    assert [m.text for m in s2.list_messages(sess.id)] == ["こんにちは", "うん。"]
    t2, _ = s2.begin_turn(sess.id, "続き", "mock", "m")
    s2.complete_turn(t2.id, [("ren", "はい。")], Usage(1, 1, 50, 5))
    assert s2.get_turn(t2.id).cache_write_tokens == 5
    s2.close()
    s3 = SQLiteStore(cast=CAST, path=path)  # 再オープンしても問題なし
    s3.close()
