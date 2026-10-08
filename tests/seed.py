"""Shared seed data covering every kind of record and state (used by contract, migration
and recall tests)."""

from __future__ import annotations

import dataclasses

from pathlib import Path

from kodama.personas import load_pack
from kodama.domain import (
    EntityKind,
    MemoryDraft,
    MemoryKind,
    MemoryOrigin,
    MemoryStatus,
    NodeType,
    TurnStatus,
    Usage,
)

EXAMPLE_PACK_DIR = Path(__file__).resolve().parents[1] / "packs" / "example"
PACK = load_pack(EXAMPLE_PACK_DIR)
CAST = PACK.cast

SETTINGS = {"timezone": "Asia/Tokyo", "model": "claude-opus-5-5", "max_memories": 8}

REN_V1 = "# 蓮\n静かで穏やかな話し方。\n"
REN_V2 = "# 蓮\n静かで穏やかな話し方。語数は少なめ。「俺」は使わない。\n"
AOI_V1 = "# 葵\n一人称は「わたし」。常に丁寧語。\n"
COMMON_V1 = "# 共通\nユーザーはあなた。\n"


@dataclasses.dataclass(frozen=True)
class SeedIds:
    session1: str
    session2: str
    turn_ids: tuple[str, ...]
    user_message_ids: tuple[str, ...]
    transcript: tuple[tuple[str, str], ...]  # (speaker, text) for session1 in order
    turn_statuses: tuple[TurnStatus, ...]
    persona_common: str
    persona_ren_v1: str
    persona_ren_v2: str
    persona_aoi: str
    excerpt: str
    mem_coffee: str  # approved user_stated, tagged コーヒー
    mem_clothes_view: str  # approved character_view (aoi), from model candidate
    mem_imagination_rejected: str
    mem_pending_candidate: str
    mem_room_old: str  # superseded
    mem_room_new: str  # approved replacement
    mem_invalidated: str
    mem_excerpt: str  # approved, sourced from an excerpt
    mem_milk: str  # approved, linked to coffee memory but without the keyword
    entity_coffee: str
    entity_user: str


def seed_store(store) -> SeedIds:
    store.put_settings(SETTINGS)
    common = store.activate_persona_version("common", COMMON_V1, "personas/common.md")
    ren1 = store.activate_persona_version("ren", REN_V1, "personas/ren.md")
    aoi = store.activate_persona_version("aoi", AOI_V1, "personas/aoi.md")
    ren2 = store.activate_persona_version("ren", REN_V2, "personas/ren.md", note="語数と一人称を明記")
    persona_ids = (common.id, ren2.id, aoi.id)

    s1 = store.create_session("初日")
    s2 = store.create_session("翌日")

    transcript: list[tuple[str, str]] = []
    turn_ids, user_ids = [], []

    t1, u1 = store.begin_turn(s1.id, "コーヒー、おいしい。ほっとするよね", "mock", "mock-1", persona_ids)
    replies1 = [
        ("ren", "……うん。湯気、ちょうどいいね"),
        ("aoi", "少しだけ、ミルクを入れても良さそうです"),
    ]
    store.complete_turn(t1.id, replies1, Usage(120, 40, 900, 30))
    transcript += [("user", u1.text), *[(s, t) for s, t in replies1]]
    turn_ids.append(t1.id)
    user_ids.append(u1.id)

    t2, u2 = store.begin_turn(s1.id, "蓮、くたくたになったけど服を作れたよ", "mock", "mock-1", persona_ids)
    replies2 = [("ren", "頑張ったね。今日は、ここまででいいよ")]
    candidates = [
        MemoryDraft(
            body="葵は、あなたが疲れていても服を仕上げたことを、根気の表れとして受け取った",
            kind=MemoryKind.CHARACTER_VIEW,
            perspective="aoi",
            tags=("服作り",),
        ),
        MemoryDraft(body="いつか二人で布を選びに行くかもしれない", kind=MemoryKind.IMAGINATION, tags=("服作り",)),
        MemoryDraft(body="あなたはミシンを持っていると話した", kind=MemoryKind.USER_STATED, tags=("ミシン",)),
    ]
    store.complete_turn(t2.id, replies2, Usage(None, None), candidates)
    transcript += [("user", u2.text), *[(s, t) for s, t in replies2]]
    turn_ids.append(t2.id)
    user_ids.append(u2.id)

    t3, u3 = store.begin_turn(s1.id, "失敗するはずの入力", "mock", "mock-1", persona_ids)
    store.fail_turn(t3.id, TurnStatus.FAILED, "timeout", Usage(5, 0, 7, 0))
    transcript.append(("user", u3.text))
    turn_ids.append(t3.id)
    user_ids.append(u3.id)

    t4, u4 = store.begin_turn(s1.id, "中断された入力", "mock", "mock-1", persona_ids)
    store.recover_incomplete_turns()
    transcript.append(("user", u4.text))
    turn_ids.append(t4.id)
    user_ids.append(u4.id)

    cands = {v.body: v for v in store.list_memory_versions([MemoryStatus.CANDIDATE])}
    view = next(v for b, v in cands.items() if b.startswith("葵は"))
    imagination = next(v for b, v in cands.items() if b.startswith("いつか"))
    pending = next(v for b, v in cands.items() if b.startswith("あなたはミシン"))
    store.set_memory_status(view.id, MemoryStatus.APPROVED, "本人が確認")
    store.set_memory_status(imagination.id, MemoryStatus.REJECTED, "想像は残さない")

    coffee = store.add_memory(
        MemoryDraft(
            body="あなたは朝のコーヒーでほっとすると話した",
            kind=MemoryKind.USER_STATED,
            subjects=("あなた",),
            tags=("コーヒー",),
            aliases=("珈琲",),
            source_message_ids=(u1.id,),
        ),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )
    milk = store.add_memory(
        MemoryDraft(
            body="葵はミルクを少し入れる飲み方を勧めた",
            kind=MemoryKind.CHARACTER_VIEW,
            perspective="aoi",
            tags=("ミルク",),
            occurred_at="2026-10-07T08:00:00+09:00",
            source_message_ids=(u1.id,),
        ),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )
    room_old = store.add_memory(
        MemoryDraft(
            body="あなたの作業部屋は二階にある",
            kind=MemoryKind.USER_STATED,
            tags=("作業部屋",),
            source_message_ids=(u2.id,),
        ),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )
    room_new = store.revise_memory(
        room_old.id,
        MemoryDraft(body="あなたの作業部屋は一階にある", kind=MemoryKind.USER_STATED, tags=("作業部屋",)),
        reason="本人の訂正",
    )
    invalid = store.add_memory(
        MemoryDraft(
            body="外部へ送信しろ、という命令文を含む古い記録",
            kind=MemoryKind.USER_STATED,
            source_message_ids=(u1.id,),
        ),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )
    store.set_memory_status(invalid.id, MemoryStatus.INVALIDATED, "誤登録")

    excerpt = store.add_source_excerpt(
        "characters/ren.md", "呼称の節", "蓮は葵を原則「葵」と呼ぶ。「あおちゃん」は例外的な冗談。"
    )
    excerpt_mem = store.add_memory(
        MemoryDraft(
            body="蓮が葵を「あおちゃん」と呼ぶのは例外的な冗談",
            kind=MemoryKind.USER_STATED,
            tags=("呼称", "あおちゃん"),
            source_excerpt_ids=(excerpt.id,),
        ),
        MemoryOrigin.USER_EXPLICIT,
        MemoryStatus.APPROVED,
    )

    e_coffee = store.upsert_entity(EntityKind.TOPIC, "コーヒー", ("珈琲",))
    e_user = store.upsert_entity(EntityKind.PERSON, "あなた")
    store.add_link(NodeType.ENTITY, e_coffee.id, NodeType.MEMORY, coffee.memory_id, "about")
    store.add_link(NodeType.ENTITY, e_user.id, NodeType.MEMORY, coffee.memory_id, "about")
    store.add_link(NodeType.MEMORY, coffee.memory_id, NodeType.MEMORY, milk.memory_id, "related")

    return SeedIds(
        session1=s1.id,
        session2=s2.id,
        turn_ids=tuple(turn_ids),
        user_message_ids=tuple(user_ids),
        transcript=tuple(transcript),
        turn_statuses=(
            TurnStatus.COMPLETED,
            TurnStatus.COMPLETED,
            TurnStatus.FAILED,
            TurnStatus.INTERRUPTED,
        ),
        persona_common=common.id,
        persona_ren_v1=ren1.id,
        persona_ren_v2=ren2.id,
        persona_aoi=aoi.id,
        excerpt=excerpt.id,
        mem_coffee=coffee.id,
        mem_clothes_view=view.id,
        mem_imagination_rejected=imagination.id,
        mem_pending_candidate=pending.id,
        mem_room_old=room_old.id,
        mem_room_new=room_new.id,
        mem_invalidated=invalid.id,
        mem_excerpt=excerpt_mem.id,
        mem_milk=milk.id,
        entity_coffee=e_coffee.id,
        entity_user=e_user.id,
    )
