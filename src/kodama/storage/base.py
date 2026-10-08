"""Store protocol: the logical storage operations used by conversation, recall and migration.

Implementations must keep every grouped update (begin_turn, complete_turn, revise_memory,
activate_persona_version, import_snapshot, ...) atomic. Nothing outside the implementation
issues SQL or depends on storage-internal identifiers.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

from kodama.domain import (
    Entity,
    EntityKind,
    ImportStats,
    Link,
    MemoryDraft,
    MemoryOrigin,
    MemoryStatus,
    MemoryVersion,
    Message,
    NodeType,
    PersonaVersion,
    Session,
    SourceExcerpt,
    Speaker,
    TranscriptEntry,
    Turn,
    TurnStatus,
    Usage,
)

# Tables/sections of a snapshot, in dependency order.
SNAPSHOT_SECTIONS = (
    "sessions",
    "persona_versions",
    "turns",
    "messages",
    "source_excerpts",
    "memory_versions",
    "entities",
    "links",
    "settings",
)


class StoreError(Exception):
    pass


class NotFound(StoreError):
    pass


class InvalidState(StoreError):
    pass


class ImportConflict(StoreError):
    """Same id already exists with different content."""

    def __init__(self, section: str, record_id: str):
        super().__init__(f"conflict in {section}: id {record_id} exists with different content")
        self.section = section
        self.record_id = record_id


class ImportValidationError(StoreError):
    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        head = "; ".join(self.errors[:5])
        more = f" (+{len(self.errors) - 5} more)" if len(self.errors) > 5 else ""
        super().__init__(f"import validation failed: {head}{more}")


class Store(Protocol):
    # sessions
    def create_session(self, title: str) -> Session: ...
    def get_session(self, session_id: str) -> Session: ...
    def list_sessions(self) -> list[Session]: ...

    # turns / messages
    def begin_turn(
        self,
        session_id: str,
        user_text: str,
        provider: str | None,
        model: str | None,
        persona_version_ids: Sequence[str] = (),
        context_memory_version_ids: Sequence[str] = (),
        context_message_ids: Sequence[str] = (),
    ) -> tuple[Turn, Message]: ...
    def complete_turn(
        self,
        turn_id: str,
        utterances: Sequence[tuple[Speaker | str, str]],
        usage: Usage | None = None,
        candidates: Sequence[MemoryDraft] = (),
        model: str | None = None,
        candidate_status: MemoryStatus = MemoryStatus.CANDIDATE,
    ) -> list[Message]: ...
    def fail_turn(
        self, turn_id: str, status: TurnStatus, error: str | None, usage: Usage | None = None
    ) -> Turn: ...
    def recover_incomplete_turns(self) -> list[Turn]: ...
    def get_turn(self, turn_id: str) -> Turn: ...
    def list_turns(self, session_id: str) -> list[Turn]: ...
    def list_messages(self, session_id: str) -> list[Message]: ...
    def recent_messages(self, session_id: str, limit: int) -> list[TranscriptEntry]: ...
    def get_message(self, message_id: str) -> Message: ...

    # personas
    def get_active_persona(self, persona_key: str) -> PersonaVersion | None: ...
    def get_persona_version(self, version_id: str) -> PersonaVersion: ...
    def list_persona_versions(self, persona_key: str | None = None) -> list[PersonaVersion]: ...
    def activate_persona_version(
        self, persona_key: str, body: str, source_path: str | None = None, note: str | None = None
    ) -> PersonaVersion: ...

    # memories
    def add_memory(
        self, draft: MemoryDraft, origin: MemoryOrigin, status: MemoryStatus = MemoryStatus.CANDIDATE
    ) -> MemoryVersion: ...
    def get_memory_version(self, version_id: str) -> MemoryVersion: ...
    def list_memory_versions(
        self, statuses: Iterable[MemoryStatus] | None = None, memory_id: str | None = None
    ) -> list[MemoryVersion]: ...
    def set_memory_status(
        self, version_id: str, new_status: MemoryStatus, reason: str | None = None
    ) -> MemoryVersion: ...
    def revise_memory(
        self,
        version_id: str,
        draft: MemoryDraft,
        reason: str | None = None,
        origin: MemoryOrigin = MemoryOrigin.USER_EXPLICIT,
    ) -> MemoryVersion: ...
    def search_memory_versions(self, terms: Sequence[str], limit: int = 50) -> list[MemoryVersion]: ...

    # entities / links
    def upsert_entity(self, kind: EntityKind, name: str, aliases: Sequence[str] = ()) -> Entity: ...
    def get_entity(self, entity_id: str) -> Entity: ...
    def list_entities(self) -> list[Entity]: ...
    def find_entities(self, terms: Sequence[str]) -> list[Entity]: ...
    def add_link(
        self,
        src_type: NodeType,
        src_id: str,
        dst_type: NodeType,
        dst_id: str,
        relation: str,
        note: str | None = None,
    ) -> Link: ...
    def links_of(self, node_type: NodeType, node_id: str, limit: int = 20) -> list[Link]: ...

    # source excerpts
    def add_source_excerpt(self, title: str, locator: str, text: str) -> SourceExcerpt: ...
    def get_source_excerpt(self, excerpt_id: str) -> SourceExcerpt: ...
    def list_source_excerpts(self) -> list[SourceExcerpt]: ...

    # settings (non-secret)
    def put_settings(self, values: dict[str, Any]) -> None: ...
    def get_settings(self) -> dict[str, Any]: ...

    # migration
    def export_snapshot(self) -> dict[str, list[dict[str, Any]]]: ...
    def import_snapshot(self, data: dict[str, list[dict[str, Any]]]) -> ImportStats: ...

    def close(self) -> None: ...
