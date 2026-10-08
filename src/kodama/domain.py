"""Logical data types for kodama (storage- and provider-independent).

All persistent entities carry an app-assigned UUIDv4 string `id`.
Timestamps are ISO 8601 strings with a UTC offset.
"""

from __future__ import annotations

import dataclasses
import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Iterable
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Asia/Tokyo"


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso(tz: str | ZoneInfo = DEFAULT_TIMEZONE) -> str:
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    return datetime.now(zone).isoformat(timespec="microseconds")


def parse_iso(value: str) -> datetime:
    """Parse an aware ISO 8601 timestamp; reject naive values."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value!r}")
    return dt


def is_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


SPEAKER_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MAX_CHARACTERS = 4


def is_speaker_id(value: object) -> bool:
    return isinstance(value, str) and SPEAKER_ID_RE.fullmatch(value) is not None


@dataclasses.dataclass(frozen=True)
class CastMember:
    id: str
    display_name: str


@dataclasses.dataclass(frozen=True)
class Cast:
    """会話の参加者。ユーザー1人とキャラクター1〜4人。IDはDBに保存される話者ID。"""

    user: CastMember
    characters: tuple[CastMember, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.characters, tuple):
            object.__setattr__(self, "characters", tuple(self.characters))
        ids = [self.user.id, *(c.id for c in self.characters)]
        for i in ids:
            if not is_speaker_id(i):
                raise ValueError(f"invalid speaker id: {i!r}")
        if not 1 <= len(self.characters) <= MAX_CHARACTERS:
            raise ValueError(f"characters must be 1..{MAX_CHARACTERS}")
        if len(set(ids)) != len(ids) or "common" in ids:
            raise ValueError("speaker ids must be unique and must not be 'common'")
        for m in (self.user, *self.characters):
            if not isinstance(m.display_name, str) or not m.display_name.strip():
                raise ValueError(f"display_name required for {m.id!r}")

    @property
    def user_id(self) -> str:
        return self.user.id

    @property
    def character_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.characters)

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return (self.user.id, *self.character_ids)

    @property
    def persona_keys(self) -> tuple[str, ...]:
        return ("common", *self.character_ids)

    def label(self, speaker_id: str) -> str:
        for m in (self.user, *self.characters):
            if m.id == speaker_id:
                return m.display_name
        return speaker_id

    def same_ids(self, other: "Cast") -> bool:
        return self.user.id == other.user.id and self.character_ids == other.character_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": {"id": self.user.id, "display_name": self.user.display_name},
            "characters": [{"id": c.id, "display_name": c.display_name} for c in self.characters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cast":
        if not isinstance(data, dict) or not isinstance(data.get("user"), dict):
            raise ValueError("cast.user is required")
        chars = data.get("characters")
        if not isinstance(chars, list):
            raise ValueError("cast.characters must be a list")
        def member(d: Any) -> CastMember:
            if not isinstance(d, dict) or set(d) != {"id", "display_name"}:
                raise ValueError(f"invalid cast member: {d!r}")
            return CastMember(id=d["id"], display_name=d["display_name"])
        return cls(user=member(data["user"]), characters=tuple(member(c) for c in chars))


class TurnStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class PersonaStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class MemoryKind(StrEnum):
    USER_STATED = "user_stated"  # 本人が話したこと・確認された設定（客観的事実ではない）
    CHARACTER_VIEW = "character_view"  # キャラクターの受け取り方（perspective 必須）
    IMAGINATION = "imagination"  # 想像・仮説


class MemoryOrigin(StrEnum):
    USER_EXPLICIT = "user_explicit"
    MODEL_CANDIDATE = "model_candidate"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class EntityKind(StrEnum):
    PERSON = "person"
    TOPIC = "topic"
    THING = "thing"


class NodeType(StrEnum):
    MEMORY = "memory"  # points at a logical memory_id
    MEMORY_VERSION = "memory_version"
    ENTITY = "entity"


class _Record:
    """Coerces enum fields and list fields (to tuples) after construction."""

    _ENUMS: ClassVar[dict[str, type[StrEnum]]] = {}
    _TUPLES: ClassVar[tuple[str, ...]] = ()
    # 旧ファイルに無くてもよい項目（欠落は None）
    _OPTIONAL: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        for name, enum_type in self._ENUMS.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        for name in self._TUPLES:
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = [v.value if isinstance(v, StrEnum) else v for v in value]
            elif isinstance(value, StrEnum):
                value = value.value
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        names = [f.name for f in dataclasses.fields(cls)]  # type: ignore[arg-type]
        missing = [n for n in names if n not in data and n not in cls._OPTIONAL]
        if missing:
            raise ValueError(f"{cls.__name__}: missing fields {missing}")
        extra = [k for k in data if k not in names]
        if extra:
            raise ValueError(f"{cls.__name__}: unknown fields {extra}")
        return cls(**{n: data.get(n) for n in names})


@dataclasses.dataclass(frozen=True)
class Session(_Record):
    id: str
    title: str
    created_at: str
    timezone: str


@dataclasses.dataclass(frozen=True)
class Turn(_Record):
    _ENUMS = {"status": TurnStatus}
    _TUPLES = ("persona_version_ids", "context_memory_version_ids", "context_message_ids")
    _OPTIONAL = ("cache_read_tokens", "cache_write_tokens")

    id: str
    session_id: str
    seq: int
    status: TurnStatus
    created_at: str
    finished_at: str | None
    error: str | None
    provider: str | None
    model: str | None
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    persona_version_ids: tuple[str, ...]
    context_memory_version_ids: tuple[str, ...]
    context_message_ids: tuple[str, ...]
    # 旧エクスポート・旧DBにはない項目。欠落は None（不明）として扱う
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class Message(_Record):
    id: str
    session_id: str
    turn_id: str
    seq: int
    speaker: str
    text: str
    created_at: str


@dataclasses.dataclass(frozen=True)
class TranscriptEntry:
    message: Message
    turn_status: TurnStatus


@dataclasses.dataclass(frozen=True)
class PersonaVersion(_Record):
    _ENUMS = {"status": PersonaStatus}

    id: str
    persona_key: str
    body: str
    content_hash: str
    source_path: str | None
    created_at: str
    status: PersonaStatus
    approved_at: str | None
    note: str | None


@dataclasses.dataclass(frozen=True)
class MemoryDraft(_Record):
    _ENUMS = {"kind": MemoryKind}
    _TUPLES = ("subjects", "tags", "aliases", "source_message_ids", "source_excerpt_ids")

    body: str
    kind: MemoryKind
    perspective: str | None = None
    subjects: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    occurred_at: str | None = None
    source_message_ids: tuple[str, ...] = ()
    source_excerpt_ids: tuple[str, ...] = ()

    def validate(self, character_ids: Iterable[str] | None = None) -> None:
        validate_memory_fields(self.body, self.kind, self.perspective, self.occurred_at, character_ids)


def validate_memory_fields(
    body: str,
    kind: MemoryKind,
    perspective: str | None,
    occurred_at: str | None,
    character_ids: Iterable[str] | None = None,
) -> None:
    """character_ids を渡すと perspective がその中にあるかも確かめる（省略時は形式のみ）。"""
    if not isinstance(body, str) or not body.strip():
        raise ValueError("memory body must be non-empty")
    if kind == MemoryKind.CHARACTER_VIEW:
        if not is_speaker_id(perspective):
            raise ValueError("character_view requires a character perspective")
        if character_ids is not None and perspective not in set(character_ids):
            raise ValueError(f"perspective {perspective!r} is not a character in the cast")
    elif perspective is not None:
        raise ValueError(f"{kind.value} must not have a perspective")
    if occurred_at is not None:
        parse_iso(occurred_at)


@dataclasses.dataclass(frozen=True)
class MemoryVersion(_Record):
    _ENUMS = {
        "kind": MemoryKind,
        "origin": MemoryOrigin,
        "status": MemoryStatus,
    }
    _TUPLES = ("subjects", "tags", "aliases", "source_message_ids", "source_excerpt_ids")

    id: str
    memory_id: str
    body: str
    kind: MemoryKind
    perspective: str | None
    subjects: tuple[str, ...]
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    occurred_at: str | None
    recorded_at: str
    origin: MemoryOrigin
    status: MemoryStatus
    source_message_ids: tuple[str, ...]
    source_excerpt_ids: tuple[str, ...]
    supersedes_version_id: str | None
    status_reason: str | None
    status_changed_at: str

    @property
    def has_source(self) -> bool:
        return bool(self.source_message_ids or self.source_excerpt_ids)


@dataclasses.dataclass(frozen=True)
class Entity(_Record):
    _ENUMS = {"kind": EntityKind}
    _TUPLES = ("aliases",)

    id: str
    kind: EntityKind
    name: str
    aliases: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Link(_Record):
    _ENUMS = {"src_type": NodeType, "dst_type": NodeType}

    id: str
    src_type: NodeType
    src_id: str
    dst_type: NodeType
    dst_id: str
    relation: str
    created_at: str
    note: str | None


@dataclasses.dataclass(frozen=True)
class SourceExcerpt(_Record):
    id: str
    title: str
    locator: str
    text: str
    approved_at: str


@dataclasses.dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class ImportStats:
    inserted: dict[str, int]
    skipped: dict[str, int]
