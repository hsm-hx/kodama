"""SQLite implementation of the Store protocol. All SQL lives in this module."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

from kodama.domain import (
    DEFAULT_TIMEZONE,
    Cast,
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
    PersonaStatus,
    PersonaVersion,
    Session,
    SourceExcerpt,
    TranscriptEntry,
    Turn,
    TurnStatus,
    Usage,
    new_id,
)
from kodama.storage.base import (
    SNAPSHOT_SECTIONS,
    ImportConflict,
    ImportValidationError,
    InvalidState,
    NotFound,
)

# 2: turns に cache_read_tokens / cache_write_tokens を追加
# 3: 話者IDを固定値の CHECK 制約から外し、cast（参加者）を schema_info に記録
SCHEMA_VERSION = 3
MAX_ERROR_CHARS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    timezone TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS persona_versions (
    id TEXT PRIMARY KEY,
    persona_key TEXT NOT NULL,
    body TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'retired')),
    approved_at TEXT,
    note TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS persona_one_active
    ON persona_versions(persona_key) WHERE status = 'active';
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed', 'interrupted')),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    provider TEXT,
    model TEXT,
    usage_input_tokens INTEGER,
    usage_output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    persona_version_ids TEXT NOT NULL,
    context_memory_version_ids TEXT NOT NULL,
    context_message_ids TEXT NOT NULL,
    UNIQUE (session_id, seq)
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn_id TEXT NOT NULL REFERENCES turns(id),
    seq INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq)
);
CREATE TABLE IF NOT EXISTS source_excerpts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    locator TEXT NOT NULL,
    text TEXT NOT NULL,
    approved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_versions (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    body TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('user_stated', 'character_view', 'imagination')),
    perspective TEXT,
    subjects TEXT NOT NULL,
    tags TEXT NOT NULL,
    aliases TEXT NOT NULL,
    occurred_at TEXT,
    recorded_at TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('user_explicit', 'model_candidate')),
    status TEXT NOT NULL
        CHECK (status IN ('candidate', 'approved', 'rejected', 'superseded', 'invalidated')),
    source_message_ids TEXT NOT NULL,
    source_excerpt_ids TEXT NOT NULL,
    supersedes_version_id TEXT REFERENCES memory_versions(id),
    status_reason TEXT,
    status_changed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_versions_memory_id ON memory_versions(memory_id);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('person', 'topic', 'thing')),
    name TEXT NOT NULL,
    aliases TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    id TEXT PRIMARY KEY,
    src_type TEXT NOT NULL CHECK (src_type IN ('memory', 'memory_version', 'entity')),
    src_id TEXT NOT NULL,
    dst_type TEXT NOT NULL CHECK (dst_type IN ('memory', 'memory_version', 'entity')),
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS links_src ON links(src_type, src_id);
CREATE INDEX IF NOT EXISTS links_dst ON links(dst_type, dst_id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# section -> (record class, json list columns, export ordering)
_TABLES: dict[str, tuple[type, tuple[str, ...], str]] = {
    "sessions": (Session, (), "created_at, id"),
    "persona_versions": (PersonaVersion, (), "persona_key, created_at, id"),
    "turns": (
        Turn,
        ("persona_version_ids", "context_memory_version_ids", "context_message_ids"),
        "session_id, seq",
    ),
    "messages": (Message, (), "session_id, seq"),
    "source_excerpts": (SourceExcerpt, (), "approved_at, id"),
    "memory_versions": (
        MemoryVersion,
        ("subjects", "tags", "aliases", "source_message_ids", "source_excerpt_ids"),
        "memory_id, recorded_at, id",
    ),
    "entities": (Entity, ("aliases",), "name, id"),
    "links": (Link, (), "created_at, id"),
}

_ALLOWED_MEMORY_TRANSITIONS = {
    (MemoryStatus.CANDIDATE, MemoryStatus.APPROVED),
    (MemoryStatus.CANDIDATE, MemoryStatus.REJECTED),
    (MemoryStatus.APPROVED, MemoryStatus.INVALIDATED),
}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _db_value(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return _dumps([str(v) for v in value])
    if isinstance(value, str):
        return str(value)  # normalises StrEnum to plain str
    return value


class SQLiteStore:
    def __init__(
        self,
        path: str | Path,
        cast: Cast | None = None,
        timezone: str = DEFAULT_TIMEZONE,
        fault_hook: Callable[[str], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.path = str(path)
        self.timezone = timezone
        self._tz = ZoneInfo(timezone)
        self.fault_hook = fault_hook
        self._clock = clock or (lambda: datetime.now(self._tz))
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()
        self.cast: Cast | None = None
        self._bind_cast(cast)

    # ------------------------------------------------------------------ infra

    def _stored_schema_version(self) -> int | None:
        has = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_info'"
        ).fetchone()
        if has is None:
            return None
        row = self._conn.execute("SELECT value FROM schema_info WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else None

    def _rebuild_without_speaker_checks(self) -> None:
        """v2以前のDBは話者IDを固定の CHECK 制約で縛っていた。表を作り直して制約を外す（データはそのまま）。"""
        self._conn.execute("PRAGMA foreign_keys = OFF")
        try:
            with self._write() as c:
                ddl = {
                    r["name"]: r["sql"]
                    for r in c.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
                }
                for table, column in (("messages", "speaker"), ("memory_versions", "perspective")):
                    old_sql = ddl.get(table) or ""
                    new_sql = _table_ddl(table)
                    if "CHECK (" + column not in old_sql and f"CHECK ({column}" not in old_sql:
                        continue
                    c.execute(new_sql.replace(f"CREATE TABLE IF NOT EXISTS {table}", f"CREATE TABLE {table}__new", 1))
                    cols = [r["name"] for r in c.execute(f"PRAGMA table_info({table})")]
                    col_list = ", ".join(cols)
                    c.execute(f"INSERT INTO {table}__new ({col_list}) SELECT {col_list} FROM {table}")
                    c.execute(f"DROP TABLE {table}")
                    c.execute(f"ALTER TABLE {table}__new RENAME TO {table}")
                violations = c.execute("PRAGMA foreign_key_check").fetchall()
                if violations:
                    raise InvalidState(f"foreign key violations after schema upgrade: {len(violations)}")
        finally:
            self._conn.execute("PRAGMA foreign_keys = ON")

    def _init_schema(self) -> None:
        version = self._stored_schema_version()
        if version is not None and version < 3:
            self._rebuild_without_speaker_checks()
        with self._write() as c:
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    c.execute(stmt)
            row = c.execute("SELECT value FROM schema_info WHERE key='schema_version'").fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO schema_info(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) in (1, 2):
                # v1 -> v2: 列の追加だけ（既存データはそのまま、新列は NULL=不明）。BEGIN IMMEDIATE 内なので原子的
                # v2 -> v3: CHECK 制約の除去は _rebuild_without_speaker_checks で済んでいる
                have = {r["name"] for r in c.execute("PRAGMA table_info(turns)")}
                for col in ("cache_read_tokens", "cache_write_tokens"):
                    if col not in have:
                        c.execute(f"ALTER TABLE turns ADD COLUMN {col} INTEGER")
                c.execute("UPDATE schema_info SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
            elif int(row["value"]) != SCHEMA_VERSION:
                raise InvalidState(
                    f"unsupported database schema_version {row['value']} (expected {SCHEMA_VERSION})"
                )

    # ------------------------------------------------------------------ cast

    def _bind_cast(self, cast: Cast | None) -> None:
        """DBに記録された cast と、渡された cast（人物設定パック）を突き合わせる。

        - 記録あり: 話者IDが一致しなければ拒否。表示名の変更は記録を更新する。
        - 記録なし（新規DB・旧DB）: 既存データの話者・視点・人物設定キーが cast に含まれることを確かめて記録する。
        """
        with self._write() as c:
            row = c.execute("SELECT value FROM schema_info WHERE key='cast'").fetchone()
            stored = Cast.from_dict(json.loads(row["value"])) if row else None
            if stored is not None:
                if cast is not None and not stored.same_ids(cast):
                    raise InvalidState(
                        "このDBの話者ID（" + ", ".join(stored.speaker_ids) + "）と、人物設定パックの話者ID（"
                        + ", ".join(cast.speaker_ids) + "）が一致しません"
                    )
                if cast is not None and cast != stored:
                    c.execute("UPDATE schema_info SET value = ? WHERE key = 'cast'", (_dumps(cast.to_dict()),))
                self.cast = cast or stored
                return
            if cast is None:
                return
            problems = _cast_mismatches(c, cast)
            if problems:
                raise InvalidState("人物設定パックの話者がこのDBのデータと合いません: " + "; ".join(problems))
            c.execute("INSERT INTO schema_info(key, value) VALUES ('cast', ?)", (_dumps(cast.to_dict()),))
            self.cast = cast

    def get_cast(self) -> Cast | None:
        return self.cast

    def _require_cast(self) -> Cast:
        if self.cast is None:
            raise InvalidState("このDBには参加者（cast）が設定されていません。人物設定パックを指定して開いてください。")
        return self.cast

    def _character_ids(self) -> tuple[str, ...] | None:
        return self.cast.character_ids if self.cast is not None else None

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            try:
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        finally:
            self._conn.execute("COMMIT")

    def _fault(self, point: str) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point)

    def _now(self) -> str:
        return self._clock().isoformat(timespec="microseconds")

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to(cls: type, row: sqlite3.Row, json_cols: tuple[str, ...]):
        data = dict(row)
        for col in json_cols:
            data[col] = tuple(json.loads(data[col]))
        return cls(**data)

    def _insert(self, c: sqlite3.Connection, section: str, record: Any) -> None:
        cls, _json_cols, _order = _TABLES[section]
        data = record.to_dict()
        cols = list(data.keys())
        c.execute(
            f"INSERT INTO {section} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            [_db_value(data[k]) for k in cols],
        )

    def _get(self, c: sqlite3.Connection, section: str, record_id: str):
        cls, json_cols, _ = _TABLES[section]
        row = c.execute(f"SELECT * FROM {section} WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise NotFound(f"{section}: {record_id}")
        return self._row_to(cls, row, json_cols)

    def _select(self, c: sqlite3.Connection, section: str, where: str = "", params: Sequence = ()):
        cls, json_cols, order = _TABLES[section]
        sql = f"SELECT * FROM {section}"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY {order}"
        return [self._row_to(cls, r, json_cols) for r in c.execute(sql, params)]

    def _next_seq(self, c: sqlite3.Connection, table: str, session_id: str) -> int:
        row = c.execute(
            f"SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM {table} WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])

    # --------------------------------------------------------------- sessions

    def create_session(self, title: str) -> Session:
        session = Session(id=new_id(), title=title, created_at=self._now(), timezone=self.timezone)
        with self._write() as c:
            self._insert(c, "sessions", session)
        return session

    def get_session(self, session_id: str) -> Session:
        with self._read() as c:
            return self._get(c, "sessions", session_id)

    def list_sessions(self) -> list[Session]:
        with self._read() as c:
            return self._select(c, "sessions")

    # ------------------------------------------------------------ turns/messages

    def begin_turn(
        self,
        session_id: str,
        user_text: str,
        provider: str | None,
        model: str | None,
        persona_version_ids: Sequence[str] = (),
        context_memory_version_ids: Sequence[str] = (),
        context_message_ids: Sequence[str] = (),
    ) -> tuple[Turn, Message]:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be non-empty")
        user_id = self._require_cast().user_id
        with self._write() as c:
            self._get(c, "sessions", session_id)
            pending = c.execute(
                "SELECT id FROM turns WHERE session_id = ? AND status = 'pending'", (session_id,)
            ).fetchone()
            if pending is not None:
                raise InvalidState(f"session has a pending turn: {pending['id']}")
            now = self._now()
            turn = Turn(
                id=new_id(),
                session_id=session_id,
                seq=self._next_seq(c, "turns", session_id),
                status=TurnStatus.PENDING,
                created_at=now,
                finished_at=None,
                error=None,
                provider=provider,
                model=model,
                usage_input_tokens=None,
                usage_output_tokens=None,
                cache_read_tokens=None,
                cache_write_tokens=None,
                persona_version_ids=tuple(persona_version_ids),
                context_memory_version_ids=tuple(context_memory_version_ids),
                context_message_ids=tuple(context_message_ids),
            )
            self._insert(c, "turns", turn)
            message = Message(
                id=new_id(),
                session_id=session_id,
                turn_id=turn.id,
                seq=self._next_seq(c, "messages", session_id),
                speaker=user_id,
                text=user_text,
                created_at=now,
            )
            self._insert(c, "messages", message)
        return turn, message

    def complete_turn(
        self,
        turn_id: str,
        utterances: Sequence[tuple[str, str]],
        usage: Usage | None = None,
        candidates: Sequence[MemoryDraft] = (),
        model: str | None = None,
        candidate_status: MemoryStatus = MemoryStatus.CANDIDATE,
    ) -> list[Message]:
        cast = self._require_cast()
        parsed: list[tuple[str, str]] = []
        for speaker, text in utterances:
            sp = str(speaker)
            if sp not in cast.character_ids:
                raise ValueError(f"invalid reply speaker: {speaker}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty utterance")
            parsed.append((sp, text))
        if not parsed:
            raise ValueError("no utterances")
        candidate_status = MemoryStatus(candidate_status)
        if candidate_status not in (MemoryStatus.CANDIDATE, MemoryStatus.APPROVED):
            raise ValueError("model memories must be saved as candidate or approved")
        for draft in candidates:
            draft.validate(cast.character_ids)

        with self._write() as c:
            turn: Turn = self._get(c, "turns", turn_id)
            if turn.status == TurnStatus.COMPLETED:
                return self._select(c, "messages", "turn_id = ? AND speaker != ?", (turn_id, cast.user_id))
            if turn.status != TurnStatus.PENDING:
                raise InvalidState(f"turn {turn_id} is {turn.status.value}")
            user_msg = c.execute(
                "SELECT id FROM messages WHERE turn_id = ? AND speaker = ?", (turn_id, cast.user_id)
            ).fetchone()
            now = self._now()
            seq = self._next_seq(c, "messages", turn.session_id)
            created: list[Message] = []
            for i, (sp, text) in enumerate(parsed):
                msg = Message(
                    id=new_id(),
                    session_id=turn.session_id,
                    turn_id=turn_id,
                    seq=seq + i,
                    speaker=sp,
                    text=text,
                    created_at=now,
                )
                self._insert(c, "messages", msg)
                created.append(msg)
            self._fault("complete_turn.after_messages")
            for draft in candidates:
                sources = draft.source_message_ids or ((user_msg["id"],) if user_msg else ())
                self._insert_memory(
                    c,
                    draft,
                    memory_id=new_id(),
                    origin=MemoryOrigin.MODEL_CANDIDATE,
                    status=candidate_status,
                    source_message_ids=tuple(sources),
                    supersedes=None,
                    reason=None,
                )
            self._fault("complete_turn.after_candidates")
            usage = usage or Usage()
            c.execute(
                "UPDATE turns SET status = 'completed', finished_at = ?, usage_input_tokens = ?,"
                " usage_output_tokens = ?, cache_read_tokens = ?, cache_write_tokens = ?,"
                " model = COALESCE(?, model) WHERE id = ?",
                (now, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
                 usage.cache_write_tokens, model, turn_id),
            )
        return created

    def fail_turn(
        self, turn_id: str, status: TurnStatus, error: str | None, usage: Usage | None = None
    ) -> Turn:
        status = TurnStatus(status)
        if status not in (TurnStatus.FAILED, TurnStatus.INTERRUPTED):
            raise ValueError("fail_turn status must be failed or interrupted")
        with self._write() as c:
            turn: Turn = self._get(c, "turns", turn_id)
            if turn.status != TurnStatus.PENDING:
                return turn
            usage = usage or Usage()
            c.execute(
                "UPDATE turns SET status = ?, finished_at = ?, error = ?,"
                " usage_input_tokens = ?, usage_output_tokens = ?,"
                " cache_read_tokens = ?, cache_write_tokens = ? WHERE id = ?",
                (
                    status.value,
                    self._now(),
                    (error or "")[:MAX_ERROR_CHARS] or None,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_tokens,
                    usage.cache_write_tokens,
                    turn_id,
                ),
            )
            return self._get(c, "turns", turn_id)

    def recover_incomplete_turns(self) -> list[Turn]:
        with self._write() as c:
            pending = self._select(c, "turns", "status = 'pending'")
            now = self._now()
            for turn in pending:
                c.execute(
                    "UPDATE turns SET status = 'interrupted', finished_at = ?, error = ? WHERE id = ?",
                    (now, "未完了のまま終了していました（再送していません）", turn.id),
                )
            return [self._get(c, "turns", t.id) for t in pending]

    def get_turn(self, turn_id: str) -> Turn:
        with self._read() as c:
            return self._get(c, "turns", turn_id)

    def list_turns(self, session_id: str) -> list[Turn]:
        with self._read() as c:
            return self._select(c, "turns", "session_id = ?", (session_id,))

    def list_messages(self, session_id: str) -> list[Message]:
        with self._read() as c:
            return self._select(c, "messages", "session_id = ?", (session_id,))

    def get_message(self, message_id: str) -> Message:
        with self._read() as c:
            return self._get(c, "messages", message_id)

    def recent_messages(self, session_id: str, limit: int) -> list[TranscriptEntry]:
        if limit <= 0:
            return []
        with self._read() as c:
            rows = c.execute(
                "SELECT m.*, t.status AS turn_status FROM messages m JOIN turns t ON t.id = m.turn_id"
                " WHERE m.session_id = ? ORDER BY m.seq DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out = []
        for r in reversed(rows):
            data = dict(r)
            status = TurnStatus(data.pop("turn_status"))
            out.append(TranscriptEntry(message=Message(**data), turn_status=status))
        return out

    # --------------------------------------------------------------- personas

    def get_active_persona(self, persona_key: str) -> PersonaVersion | None:
        with self._read() as c:
            found = self._select(c, "persona_versions", "persona_key = ? AND status = 'active'", (persona_key,))
            return found[0] if found else None

    def get_persona_version(self, version_id: str) -> PersonaVersion:
        with self._read() as c:
            return self._get(c, "persona_versions", version_id)

    def list_persona_versions(self, persona_key: str | None = None) -> list[PersonaVersion]:
        with self._read() as c:
            if persona_key is None:
                return self._select(c, "persona_versions")
            return self._select(c, "persona_versions", "persona_key = ?", (persona_key,))

    def activate_persona_version(
        self, persona_key: str, body: str, source_path: str | None = None, note: str | None = None
    ) -> PersonaVersion:
        if not body.strip():
            raise ValueError("persona body must be non-empty")
        if self.cast is not None and persona_key not in self.cast.persona_keys:
            raise ValueError(f"unknown persona key: {persona_key}")
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        with self._write() as c:
            active = self._select(
                c, "persona_versions", "persona_key = ? AND status = 'active'", (persona_key,)
            )
            if active and active[0].content_hash == content_hash:
                return active[0]
            now = self._now()
            c.execute(
                "UPDATE persona_versions SET status = 'retired' WHERE persona_key = ? AND status = 'active'",
                (persona_key,),
            )
            version = PersonaVersion(
                id=new_id(),
                persona_key=persona_key,
                body=body,
                content_hash=content_hash,
                source_path=source_path,
                created_at=now,
                status=PersonaStatus.ACTIVE,
                approved_at=now,
                note=note,
            )
            self._insert(c, "persona_versions", version)
            return version

    # --------------------------------------------------------------- memories

    def _check_sources(
        self, c: sqlite3.Connection, message_ids: Iterable[str], excerpt_ids: Iterable[str]
    ) -> None:
        for mid in message_ids:
            self._get(c, "messages", mid)
        for eid in excerpt_ids:
            self._get(c, "source_excerpts", eid)

    def _insert_memory(
        self,
        c: sqlite3.Connection,
        draft: MemoryDraft,
        *,
        memory_id: str,
        origin: MemoryOrigin,
        status: MemoryStatus,
        source_message_ids: tuple[str, ...],
        supersedes: str | None,
        reason: str | None,
        source_excerpt_ids: tuple[str, ...] | None = None,
    ) -> MemoryVersion:
        draft.validate(self._character_ids())
        excerpts = draft.source_excerpt_ids if source_excerpt_ids is None else source_excerpt_ids
        self._check_sources(c, source_message_ids, excerpts)
        now = self._now()
        version = MemoryVersion(
            id=new_id(),
            memory_id=memory_id,
            body=draft.body,
            kind=draft.kind,
            perspective=draft.perspective,
            subjects=draft.subjects,
            tags=draft.tags,
            aliases=draft.aliases,
            occurred_at=draft.occurred_at,
            recorded_at=now,
            origin=origin,
            status=status,
            source_message_ids=source_message_ids,
            source_excerpt_ids=excerpts,
            supersedes_version_id=supersedes,
            status_reason=reason,
            status_changed_at=now,
        )
        self._insert(c, "memory_versions", version)
        return version

    def add_memory(
        self, draft: MemoryDraft, origin: MemoryOrigin, status: MemoryStatus = MemoryStatus.CANDIDATE
    ) -> MemoryVersion:
        status = MemoryStatus(status)
        if status not in (MemoryStatus.CANDIDATE, MemoryStatus.APPROVED):
            raise ValueError("new memories must be candidate or approved")
        with self._write() as c:
            return self._insert_memory(
                c,
                draft,
                memory_id=new_id(),
                origin=MemoryOrigin(origin),
                status=status,
                source_message_ids=draft.source_message_ids,
                supersedes=None,
                reason=None,
            )

    def get_memory_version(self, version_id: str) -> MemoryVersion:
        with self._read() as c:
            return self._get(c, "memory_versions", version_id)

    def list_memory_versions(
        self, statuses: Iterable[MemoryStatus] | None = None, memory_id: str | None = None
    ) -> list[MemoryVersion]:
        clauses, params = [], []
        if statuses is not None:
            sts = [MemoryStatus(s).value for s in statuses]
            if not sts:
                return []
            clauses.append(f"status IN ({', '.join('?' for _ in sts)})")
            params.extend(sts)
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        with self._read() as c:
            return self._select(c, "memory_versions", " AND ".join(clauses), params)

    def set_memory_status(
        self, version_id: str, new_status: MemoryStatus, reason: str | None = None
    ) -> MemoryVersion:
        new_status = MemoryStatus(new_status)
        with self._write() as c:
            current: MemoryVersion = self._get(c, "memory_versions", version_id)
            if (current.status, new_status) not in _ALLOWED_MEMORY_TRANSITIONS:
                raise InvalidState(f"cannot change memory {current.status.value} -> {new_status.value}")
            c.execute(
                "UPDATE memory_versions SET status = ?, status_reason = ?, status_changed_at = ? WHERE id = ?",
                (new_status.value, reason, self._now(), version_id),
            )
            return self._get(c, "memory_versions", version_id)

    def revise_memory(
        self,
        version_id: str,
        draft: MemoryDraft,
        reason: str | None = None,
        origin: MemoryOrigin = MemoryOrigin.USER_EXPLICIT,
    ) -> MemoryVersion:
        draft.validate(self._character_ids())
        with self._write() as c:
            old: MemoryVersion = self._get(c, "memory_versions", version_id)
            if old.status not in (MemoryStatus.APPROVED, MemoryStatus.CANDIDATE):
                raise InvalidState(f"cannot revise a {old.status.value} memory")
            msg_ids = draft.source_message_ids or old.source_message_ids
            excerpt_ids = draft.source_excerpt_ids or old.source_excerpt_ids
            new = self._insert_memory(
                c,
                draft,
                memory_id=old.memory_id,
                origin=MemoryOrigin(origin),
                status=MemoryStatus.APPROVED,
                source_message_ids=tuple(msg_ids),
                source_excerpt_ids=tuple(excerpt_ids),
                supersedes=old.id,
                reason=reason,
            )
            self._fault("revise_memory.after_insert")
            c.execute(
                "UPDATE memory_versions SET status = 'superseded', status_reason = ?, status_changed_at = ?"
                " WHERE id = ?",
                (reason, new.recorded_at, old.id),
            )
            self._fault("revise_memory.after_supersede")
            self._insert(
                c,
                "links",
                Link(
                    id=new_id(),
                    src_type=NodeType.MEMORY_VERSION,
                    src_id=new.id,
                    dst_type=NodeType.MEMORY_VERSION,
                    dst_id=old.id,
                    relation="supersedes",
                    created_at=new.recorded_at,
                    note=reason,
                ),
            )
            return new

    def search_memory_versions(self, terms: Sequence[str], limit: int = 50) -> list[MemoryVersion]:
        terms = [t for t in (s.strip() for s in terms) if t]
        if not terms or limit <= 0:
            return []
        hay = "(body || ' ' || tags || ' ' || aliases || ' ' || subjects)"
        where = " OR ".join(f"{hay} LIKE ? ESCAPE '\\'" for _ in terms)
        params: list[Any] = [f"%{_escape_like(t)}%" for t in terms]
        cls, json_cols, _ = _TABLES["memory_versions"]
        with self._read() as c:
            rows = c.execute(
                f"SELECT * FROM memory_versions WHERE {where} ORDER BY recorded_at DESC, id LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [self._row_to(cls, r, json_cols) for r in rows]

    # -------------------------------------------------------- entities / links

    def upsert_entity(self, kind: EntityKind, name: str, aliases: Sequence[str] = ()) -> Entity:
        kind = EntityKind(kind)
        if not name.strip():
            raise ValueError("entity name must be non-empty")
        with self._write() as c:
            found = self._select(c, "entities", "kind = ? AND name = ?", (kind.value, name))
            if found:
                existing: Entity = found[0]
                merged = tuple(dict.fromkeys([*existing.aliases, *aliases]))
                if merged != existing.aliases:
                    c.execute("UPDATE entities SET aliases = ? WHERE id = ?", (_db_value(merged), existing.id))
                return self._get(c, "entities", existing.id)
            entity = Entity(id=new_id(), kind=kind, name=name, aliases=tuple(dict.fromkeys(aliases)))
            self._insert(c, "entities", entity)
            return entity

    def get_entity(self, entity_id: str) -> Entity:
        with self._read() as c:
            return self._get(c, "entities", entity_id)

    def list_entities(self) -> list[Entity]:
        with self._read() as c:
            return self._select(c, "entities")

    def find_entities(self, terms: Sequence[str]) -> list[Entity]:
        terms = [t for t in (s.strip() for s in terms) if t]
        if not terms:
            return []
        out = []
        for entity in self.list_entities():
            names = [entity.name, *entity.aliases]
            if any(n and (n in t or t in n) for n in names for t in terms):
                out.append(entity)
        return out

    def _check_node(self, c: sqlite3.Connection, node_type: NodeType, node_id: str) -> None:
        if node_type == NodeType.ENTITY:
            self._get(c, "entities", node_id)
        elif node_type == NodeType.MEMORY_VERSION:
            self._get(c, "memory_versions", node_id)
        elif node_type == NodeType.MEMORY:
            if c.execute("SELECT 1 FROM memory_versions WHERE memory_id = ?", (node_id,)).fetchone() is None:
                raise NotFound(f"memory: {node_id}")

    def add_link(
        self,
        src_type: NodeType,
        src_id: str,
        dst_type: NodeType,
        dst_id: str,
        relation: str,
        note: str | None = None,
    ) -> Link:
        src_type, dst_type = NodeType(src_type), NodeType(dst_type)
        if not relation.strip():
            raise ValueError("relation must be non-empty")
        with self._write() as c:
            self._check_node(c, src_type, src_id)
            self._check_node(c, dst_type, dst_id)
            found = self._select(
                c,
                "links",
                "src_type = ? AND src_id = ? AND dst_type = ? AND dst_id = ? AND relation = ?",
                (src_type.value, src_id, dst_type.value, dst_id, relation),
            )
            if found:
                return found[0]
            link = Link(
                id=new_id(),
                src_type=src_type,
                src_id=src_id,
                dst_type=dst_type,
                dst_id=dst_id,
                relation=relation,
                created_at=self._now(),
                note=note,
            )
            self._insert(c, "links", link)
            return link

    def links_of(self, node_type: NodeType, node_id: str, limit: int = 20) -> list[Link]:
        if limit <= 0:
            return []
        node_type = NodeType(node_type)
        cls, json_cols, _ = _TABLES["links"]
        with self._read() as c:
            rows = c.execute(
                "SELECT * FROM links WHERE (src_type = ? AND src_id = ?) OR (dst_type = ? AND dst_id = ?)"
                " ORDER BY created_at, id LIMIT ?",
                (node_type.value, node_id, node_type.value, node_id, limit),
            ).fetchall()
        return [self._row_to(cls, r, json_cols) for r in rows]

    # -------------------------------------------------------- source excerpts

    def add_source_excerpt(self, title: str, locator: str, text: str) -> SourceExcerpt:
        if not text.strip():
            raise ValueError("excerpt text must be non-empty")
        excerpt = SourceExcerpt(id=new_id(), title=title, locator=locator, text=text, approved_at=self._now())
        with self._write() as c:
            self._insert(c, "source_excerpts", excerpt)
        return excerpt

    def get_source_excerpt(self, excerpt_id: str) -> SourceExcerpt:
        with self._read() as c:
            return self._get(c, "source_excerpts", excerpt_id)

    def list_source_excerpts(self) -> list[SourceExcerpt]:
        with self._read() as c:
            return self._select(c, "source_excerpts")

    # ---------------------------------------------------------------- settings

    def put_settings(self, values: dict[str, Any]) -> None:
        with self._write() as c:
            for key, value in values.items():
                c.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, _dumps(value)),
                )

    def get_settings(self) -> dict[str, Any]:
        with self._read() as c:
            return {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT * FROM settings ORDER BY key")}

    # --------------------------------------------------------------- migration

    def export_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        with self._read() as c:  # one read transaction = consistent snapshot
            for section in SNAPSHOT_SECTIONS:
                if section == "settings":
                    out[section] = [
                        {"key": r["key"], "value": json.loads(r["value"])}
                        for r in c.execute("SELECT * FROM settings ORDER BY key")
                    ]
                else:
                    out[section] = [rec.to_dict() for rec in self._select(c, section)]
        return out

    def import_snapshot(self, data: dict[str, list[dict[str, Any]]]) -> ImportStats:
        missing = [s for s in SNAPSHOT_SECTIONS if s not in data]
        if missing:
            raise ImportValidationError([f"missing sections: {missing}"])
        # Build records first so malformed input fails before any write.
        records: dict[str, list[Any]] = {}
        errors: list[str] = []
        for section in SNAPSHOT_SECTIONS:
            records[section] = []
            for i, raw in enumerate(data[section]):
                try:
                    if section == "settings":
                        if set(raw) != {"key", "value"} or not isinstance(raw["key"], str):
                            raise ValueError("settings entries need exactly key and value")
                        records[section].append(dict(raw))
                    else:
                        records[section].append(_TABLES[section][0].from_dict(raw))
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{section}[{i}]: {exc}")
        if self.cast is not None:
            errors += _snapshot_cast_errors(records, self.cast)
        if errors:
            raise ImportValidationError(errors)

        inserted = {s: 0 for s in SNAPSHOT_SECTIONS}
        skipped = {s: 0 for s in SNAPSHOT_SECTIONS}
        try:
            with self._write() as c:
                c.execute("PRAGMA defer_foreign_keys = ON")
                for section in SNAPSHOT_SECTIONS:
                    for rec in records[section]:
                        if section == "settings":
                            row = c.execute("SELECT value FROM settings WHERE key = ?", (rec["key"],)).fetchone()
                            if row is not None:
                                if json.loads(row["value"]) != rec["value"]:
                                    raise ImportConflict(section, rec["key"])
                                skipped[section] += 1
                                continue
                            c.execute(
                                "INSERT INTO settings(key, value) VALUES (?, ?)",
                                (rec["key"], _dumps(rec["value"])),
                            )
                        else:
                            try:
                                existing = self._get(c, section, rec.id)
                            except NotFound:
                                existing = None
                            if existing is not None:
                                if existing.to_dict() != rec.to_dict():
                                    raise ImportConflict(section, rec.id)
                                skipped[section] += 1
                                continue
                            self._insert(c, section, rec)
                        inserted[section] += 1
                    self._fault(f"import_snapshot.after_{section}")
                violations = c.execute("PRAGMA foreign_key_check").fetchall()
                if violations:
                    raise ImportValidationError(
                        [f"broken reference in {v[0]} (parent {v[2]})" for v in violations]
                    )
        except sqlite3.IntegrityError as exc:
            raise ImportValidationError([f"integrity error: {exc}"]) from exc
        return ImportStats(inserted=inserted, skipped=skipped)


def _table_ddl(table: str) -> str:
    for stmt in _SCHEMA.split(";"):
        if stmt.strip().startswith(f"CREATE TABLE IF NOT EXISTS {table} ("):
            return stmt.strip()
    raise KeyError(table)


def _cast_mismatches(c: sqlite3.Connection, cast: Cast) -> list[str]:
    problems: list[str] = []
    speakers = {r[0] for r in c.execute("SELECT DISTINCT speaker FROM messages")}
    if extra := sorted(speakers - set(cast.speaker_ids)):
        problems.append(f"messages.speaker {extra}")
    persp = {r[0] for r in c.execute("SELECT DISTINCT perspective FROM memory_versions WHERE perspective IS NOT NULL")}
    if extra := sorted(persp - set(cast.character_ids)):
        problems.append(f"memory_versions.perspective {extra}")
    keys = {r[0] for r in c.execute("SELECT DISTINCT persona_key FROM persona_versions")}
    if extra := sorted(keys - set(cast.persona_keys)):
        problems.append(f"persona_versions.persona_key {extra}")
    return problems


def _snapshot_cast_errors(records: dict[str, list[Any]], cast: Cast) -> list[str]:
    errors: list[str] = []
    for m in records.get("messages", []):
        if m.speaker not in cast.speaker_ids:
            errors.append(f"messages {m.id}: speaker {m.speaker!r} is not in the cast")
    for v in records.get("memory_versions", []):
        if v.perspective is not None and v.perspective not in cast.character_ids:
            errors.append(f"memory_versions {v.id}: perspective {v.perspective!r} is not a character")
    for p in records.get("persona_versions", []):
        if p.persona_key not in cast.persona_keys:
            errors.append(f"persona_versions {p.id}: persona_key {p.persona_key!r} is not in the cast")
    return errors


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
