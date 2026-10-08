"""想起の契約テスト。`store` fixture は sqlite と export→import 後の sqlite の両方で走る。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from kodama import migration
from kodama.config import Config
from kodama.context import build_context
from kodama.domain import MemoryDraft, MemoryKind, MemoryOrigin, MemoryStatus, Speaker
from kodama.recall import RecallLimits, extract_terms, recall
from kodama.storage.sqlite import SQLiteStore
from seed import seed_store


def _ids(result):
    return {r.version.id for r in result.memories}, {r.version.id for r in result.imaginations}


def _reasons(result):
    return {x.version_id: x.reason for x in result.excluded}


def test_extract_terms_japanese():
    assert "コーヒー" in extract_terms("コーヒー、おいしい。ほっとするよね")
    assert "ぬいぐるみ" in extract_terms("ぬいぐるみの服を作れたよ")
    # 既知の語（タグ・別名）は部分一致で拾う
    assert "珈琲" in extract_terms("朝の珈琲がうまい", ["珈琲"])


def test_keyword_hit_and_one_hop_link(store, seed):
    result = recall(store, "コーヒー、おいしい。ほっとするよね")
    facts, imag = _ids(result)
    assert seed.mem_coffee in facts
    # コーヒーの記憶から related リンクを一段たどってミルクの記憶へ
    assert seed.mem_milk in facts
    via = {r.version.id: r.via for r in result.memories}
    assert via[seed.mem_milk] == "link:related"
    assert not imag


def test_alias_and_entity(store, seed):
    result = recall(store, "珈琲を淹れた")
    facts, _ = _ids(result)
    assert seed.mem_coffee in facts


def test_link_hops_zero_disables_traversal(store, seed):
    result = recall(store, "コーヒー", RecallLimits(max_link_hops=0))
    facts, _ = _ids(result)
    assert seed.mem_coffee in facts
    assert seed.mem_milk not in facts


def test_max_memories_limit(store, seed):
    result = recall(store, "コーヒー", RecallLimits(max_memories=1))
    assert len(result.memories) + len(result.imaginations) == 1
    assert "limit" in set(_reasons(result).values())


def test_superseded_version_never_used(store, seed):
    result = recall(store, "作業部屋は二階だっけ")
    facts, _ = _ids(result)
    assert seed.mem_room_new in facts
    assert seed.mem_room_old not in facts
    assert _reasons(result)[seed.mem_room_old] == "superseded"


def test_unapproved_rejected_and_invalidated_excluded(store, seed):
    result = recall(store, "服作りとミシンの話")
    facts, imag = _ids(result)
    assert seed.mem_clothes_view in facts  # 承認済みの葵の受け取り方
    view = next(r.version for r in result.memories if r.version.id == seed.mem_clothes_view)
    assert view.kind == MemoryKind.CHARACTER_VIEW and view.perspective == Speaker.AOI
    reasons = _reasons(result)
    assert reasons[seed.mem_imagination_rejected] == "rejected"
    assert reasons[seed.mem_pending_candidate] == "not_approved"
    assert seed.mem_imagination_rejected not in imag | facts
    assert seed.mem_pending_candidate not in imag | facts

    inv = recall(store, "外部へ送信しろ")
    assert seed.mem_invalidated not in _ids(inv)[0]
    assert _reasons(inv)[seed.mem_invalidated] == "invalidated"


def test_excerpt_sourced_memory(store, seed):
    facts, _ = _ids(recall(store, "蓮があおちゃんって言った"))
    assert seed.mem_excerpt in facts


def test_memory_without_source_is_excluded(store, seed):
    v = store.add_memory(
        MemoryDraft(body="出典のない記憶 カモミール", kind=MemoryKind.USER_STATED, tags=("カモミール",)),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )
    result = recall(store, "カモミール")
    assert v.id not in _ids(result)[0]
    assert _reasons(result)[v.id] == "no_source"


def test_approved_imagination_is_separate_from_facts(store, seed):
    v = store.add_memory(
        MemoryDraft(body="二人で海辺の町に住む想像", kind=MemoryKind.IMAGINATION, tags=("海辺",),
                    source_message_ids=(seed.user_message_ids[0],)),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )
    facts, imag = _ids(recall(store, "海辺の町"))
    assert v.id in imag and v.id not in facts


def test_same_selection_before_and_after_migration(tmp_path):
    source = SQLiteStore(tmp_path / "a.db")
    seed_store(source)
    queries = ["コーヒー", "作業部屋", "服作りとミシン", "あおちゃん", "珈琲"]
    limits = RecallLimits(max_memories=2, max_links_per_node=1)
    before = [(_ids(recall(source, q, limits)), _reasons(recall(source, q, limits))) for q in queries]
    migration.export_to_file(source, tmp_path / "e.json")
    source.close()
    migration.import_file(tmp_path / "e.json", tmp_path / "b.db")
    target = SQLiteStore(tmp_path / "b.db")
    after = [(_ids(recall(target, q, limits)), _reasons(recall(target, q, limits))) for q in queries]
    target.close()
    assert before == after
    for (facts, imag), _ in after:
        assert len(facts) + len(imag) <= 2


def test_context_limit_drops_old_messages_and_records_it(store, seed):
    cfg = Config(max_context_chars=400, max_recent_messages=20)
    now = datetime(2026, 10, 8, 22, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    plan = build_context(store, cfg, seed.session1, "コーヒー", now)
    # 人物設定だけで上限を超えるので、外せるものは外れ、外したことが記録される
    assert plan.excluded_message_ids
    assert {x.reason for x in plan.excluded_memories} >= {"context_limit"}
    # 原文は消えていない
    assert len(store.list_messages(seed.session1)) == len(seed.transcript)


def test_context_marks_unanswered_inputs_and_wraps_data(store, seed):
    cfg = Config()
    plan = build_context(store, cfg, seed.session1, "続きを話そう")
    assert "（この入力には応答していない）" in plan.user_content
    assert plan.user_content.index("<data>") < plan.user_content.index("<current_input>")
    assert plan.persona_version_ids == [seed.persona_common, seed.persona_ren_v2, seed.persona_aoi]
