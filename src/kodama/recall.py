"""手掛かりからの想起（design.md §5）。Store Protocol だけを使い、保存方式に依存しない。

採用条件・除外理由・上限はここで決める。保存先が変わってもこの契約は変えない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from kodama.domain import MemoryKind, MemoryStatus, MemoryVersion, NodeType

# 語の区切りに使う記号と、簡易的な助詞・語尾（形態素解析はしない）
_SPLIT_RE = re.compile(r"[\s、。，．,.!！?？「」『』（）()\[\]【】…・〜~―:：;；\"'“”‘’/]+")
_PARTICLE_RE = re.compile(
    r"(?:から|まで|より|って|けど|ので|のに|では|には|とは|でも|だよ|です|ます|でした|ました|"
    r"は|が|を|に|で|と|も|の|へ|や|ね|よ|か|な)"
)
_HIRAGANA_ONLY = re.compile(r"^[ぁ-ゟ]+$")
MAX_TERMS = 16

_STATUS_REASON = {
    MemoryStatus.CANDIDATE: "not_approved",
    MemoryStatus.REJECTED: "rejected",
    MemoryStatus.SUPERSEDED: "superseded",
    MemoryStatus.INVALIDATED: "invalidated",
}

REASON_LABELS = {
    "not_approved": "未承認の候補",
    "rejected": "却下済み",
    "superseded": "訂正済みの旧版",
    "invalidated": "無効化済み",
    "no_source": "出典がない",
    "limit": "件数上限",
    "context_limit": "コンテキスト量の上限",
}


@dataclass(frozen=True)
class RecallLimits:
    max_memories: int = 8
    max_candidates_scanned: int = 50
    max_link_hops: int = 1
    max_links_per_node: int = 10


@dataclass(frozen=True)
class RecalledMemory:
    version: MemoryVersion
    via: str  # 例: "keyword:コーヒー", "entity:コーヒー", "link:related", "correction"


@dataclass(frozen=True)
class ExcludedMemory:
    version_id: str
    memory_id: str
    reason: str


@dataclass
class RecallResult:
    terms: list[str]
    memories: list[RecalledMemory] = field(default_factory=list)  # user_stated / character_view
    imaginations: list[RecalledMemory] = field(default_factory=list)  # 想像・仮説（事実と分ける）
    excluded: list[ExcludedMemory] = field(default_factory=list)
    scanned: int = 0

    @property
    def used_version_ids(self) -> list[str]:
        return [r.version.id for r in (*self.memories, *self.imaginations)]


def _chunks(text: str) -> list[str]:
    out: list[str] = []
    for piece in _SPLIT_RE.split(text):
        for chunk in _PARTICLE_RE.split(piece):
            chunk = chunk.strip()
            if len(chunk) < 2 or _HIRAGANA_ONLY.match(chunk) and len(chunk) < 3:
                continue
            out.append(chunk)
    return out


def extract_terms(text: str, vocabulary: Sequence[str] = ()) -> list[str]:
    """入力から検索語を取り出す。既知の語（タグ・別名・実体名）の部分一致を優先する。"""
    terms: list[str] = []
    for word in sorted({v for v in vocabulary if v and len(v) >= 2}, key=len, reverse=True):
        if word in text:
            terms.append(word)
    terms.extend(_chunks(text))
    seen: dict[str, None] = {}
    for t in terms:
        seen.setdefault(t, None)
    return list(seen)[:MAX_TERMS]


def _vocabulary(store) -> list[str]:
    vocab: list[str] = []
    for v in store.list_memory_versions([MemoryStatus.APPROVED]):
        vocab.extend(v.tags)
        vocab.extend(v.aliases)
        vocab.extend(v.subjects)
    for e in store.list_entities():
        vocab.append(e.name)
        vocab.extend(e.aliases)
    return vocab


def _current_version(store, memory_id: str) -> MemoryVersion | None:
    versions = store.list_memory_versions(memory_id=memory_id)
    approved = [v for v in versions if v.status == MemoryStatus.APPROVED]
    if not approved:
        return None
    superseded_ids = {v.supersedes_version_id for v in versions if v.supersedes_version_id}
    live = [v for v in approved if v.id not in superseded_ids]
    return max(live, key=lambda v: v.recorded_at) if live else None


def recall(store, query_text: str, limits: RecallLimits = RecallLimits()) -> RecallResult:
    terms = extract_terms(query_text, _vocabulary(store))
    result = RecallResult(terms=terms)
    if not terms:
        return result

    # (version, via) を優先度順に集める
    ordered: list[tuple[MemoryVersion, str]] = []
    hits = store.search_memory_versions(terms, limit=limits.max_candidates_scanned)
    result.scanned = len(hits)

    def score(v: MemoryVersion) -> int:
        hay = " ".join([v.body, *v.tags, *v.aliases, *v.subjects])
        return sum(1 for t in terms if t in hay)

    newest_first = sorted(hits, key=lambda v: v.recorded_at, reverse=True)
    for v in sorted(newest_first, key=lambda v: -score(v)):
        matched = next((t for t in terms if t in " ".join([v.body, *v.tags, *v.aliases, *v.subjects])), "")
        ordered.append((v, f"keyword:{matched}"))

    seed_memory_ids: list[str] = []
    if limits.max_link_hops >= 1:
        for entity in store.find_entities(terms)[: limits.max_links_per_node]:
            for link in store.links_of(NodeType.ENTITY, entity.id, limit=limits.max_links_per_node):
                if link.relation != "about":
                    continue
                other = _other_end(link, NodeType.ENTITY, entity.id)
                if other and other[0] == NodeType.MEMORY:
                    cur = _current_version(store, other[1])
                    if cur is not None:
                        ordered.append((cur, f"entity:{entity.name}"))

    accepted: list[RecalledMemory] = []
    accepted_memory_ids: set[str] = set()
    excluded: dict[str, ExcludedMemory] = {}

    def consider(v: MemoryVersion, via: str) -> None:
        if v.status != MemoryStatus.APPROVED:
            excluded.setdefault(v.id, ExcludedMemory(v.id, v.memory_id, _STATUS_REASON[v.status]))
            if v.status == MemoryStatus.SUPERSEDED and v.memory_id not in accepted_memory_ids:
                cur = _current_version(store, v.memory_id)
                if cur is not None:
                    consider(cur, "correction")
            return
        if v.memory_id in accepted_memory_ids:
            return
        cur = _current_version(store, v.memory_id)
        if cur is None or cur.id != v.id:
            excluded.setdefault(v.id, ExcludedMemory(v.id, v.memory_id, "superseded"))
            return
        if not v.has_source:
            excluded.setdefault(v.id, ExcludedMemory(v.id, v.memory_id, "no_source"))
            return
        accepted.append(RecalledMemory(v, via))
        accepted_memory_ids.add(v.memory_id)
        seed_memory_ids.append(v.memory_id)

    for v, via in ordered:
        consider(v, via)

    # 採用した記憶から関連リンクを一段だけたどる（たどった先からはさらにたどらない）
    if limits.max_link_hops >= 1:
        for mid in list(seed_memory_ids):
            for link in store.links_of(NodeType.MEMORY, mid, limit=limits.max_links_per_node):
                if link.relation not in ("related", "about"):
                    continue
                other = _other_end(link, NodeType.MEMORY, mid)
                if not other or other[0] != NodeType.MEMORY or other[1] in accepted_memory_ids:
                    continue
                cur = _current_version(store, other[1])
                if cur is None:
                    excluded.setdefault(other[1], ExcludedMemory("", other[1], "not_approved"))  # 有効な承認版がない
                    continue
                consider(cur, "link:related")

    for r in accepted:
        bucket = result.imaginations if r.version.kind == MemoryKind.IMAGINATION else result.memories
        total = len(result.memories) + len(result.imaginations)
        if total >= limits.max_memories:
            excluded.setdefault(r.version.id, ExcludedMemory(r.version.id, r.version.memory_id, "limit"))
            continue
        bucket.append(r)
    result.excluded = list(excluded.values())
    return result


def _other_end(link, node_type: NodeType, node_id: str) -> tuple[NodeType, str] | None:
    if link.src_type == node_type and link.src_id == node_id:
        return link.dst_type, link.dst_id
    if link.dst_type == node_type and link.dst_id == node_id:
        return link.src_type, link.src_id
    return None
