"""Versioned export/import of the whole store. Never calls a model or the network."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from kodama import __version__
from kodama.domain import (
    Entity,
    Link,
    MemoryStatus,
    MemoryVersion,
    Message,
    NodeType,
    PersonaStatus,
    PersonaVersion,
    Session,
    SourceExcerpt,
    Turn,
    is_uuid,
    parse_iso,
    validate_memory_fields,
)
from kodama.storage.base import SNAPSHOT_SECTIONS, ImportValidationError, StoreError
from kodama.storage.sqlite import SQLiteStore

FORMAT = "kodama-export"
SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1,)

# Only these (non-secret) settings are exported.
SETTINGS_ALLOWLIST = frozenset(
    {
        "timezone",
        "provider",
        "model",
        "max_tokens",
        "timeout_seconds",
        "effort",
        "max_recent_messages",
        "max_memories",
        "max_candidates_scanned",
        "max_context_chars",
        "max_link_hops",
        "max_links_per_node",
        "memory_auto_approve",
        "show_memory_notices",
    }
)

_RECORD_TYPES: dict[str, type] = {
    "sessions": Session,
    "persona_versions": PersonaVersion,
    "turns": Turn,
    "messages": Message,
    "source_excerpts": SourceExcerpt,
    "memory_versions": MemoryVersion,
    "entities": Entity,
    "links": Link,
}
_TIMESTAMP_FIELDS = {
    "sessions": ("created_at",),
    "persona_versions": ("created_at", "approved_at"),
    "turns": ("created_at", "finished_at"),
    "messages": ("created_at",),
    "source_excerpts": ("approved_at",),
    "memory_versions": ("occurred_at", "recorded_at", "status_changed_at"),
    "links": ("created_at",),
}
_REQUIRED_TEXT = {
    "sessions": ("title", "timezone"),
    "persona_versions": ("persona_key", "body", "content_hash"),
    "messages": ("text",),
    "source_excerpts": ("title", "locator", "text"),
    "memory_versions": ("body",),
    "entities": ("name",),
    "links": ("relation",),
}


@dataclasses.dataclass
class VerifyReport:
    ok: bool
    errors: list[str]
    schema_version: Any = None
    counts: dict[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ImportResult:
    target: str
    created_new: bool
    inserted: dict[str, int]
    skipped: dict[str, int]


def _looks_secret(key: str, value: Any) -> bool:
    lowered = key.lower()
    if any(word in lowered for word in ("key", "token", "secret", "password")):
        return True
    return isinstance(value, str) and value.startswith(("sk-", "sk_"))


def filter_settings(values: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in values.items() if k in SETTINGS_ALLOWLIST and not _looks_secret(k, v)
    }


def checksum(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_export(store, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    data = store.export_snapshot()
    merged = filter_settings({row["key"]: row["value"] for row in data["settings"]})
    merged.update(filter_settings(settings or {}))
    data["settings"] = [{"key": k, "value": merged[k]} for k in sorted(merged)]
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "exported_at": store._now() if hasattr(store, "_now") else None,
        "app_version": __version__,
        "counts": {s: len(data[s]) for s in SNAPSHOT_SECTIONS},
        "data": data,
        "checksum": checksum(data),
    }


def export_to_file(store, path: str | Path, settings: dict[str, Any] | None = None) -> dict[str, int]:
    doc = build_export(store, settings)
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return doc["counts"]


# ------------------------------------------------------------------ verification


def verify_document(doc: Any) -> VerifyReport:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return VerifyReport(False, ["top level is not an object"])
    version = doc.get("schema_version")
    if doc.get("format") != FORMAT:
        errors.append(f"unknown format: {doc.get('format')!r}")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"unsupported schema_version: {version!r}")
        return VerifyReport(False, errors, version)
    data = doc.get("data")
    if not isinstance(data, dict):
        return VerifyReport(False, errors + ["data is missing"], version)
    for section in SNAPSHOT_SECTIONS:
        if not isinstance(data.get(section), list):
            errors.append(f"section {section} missing or not a list")
    extra = set(data) - set(SNAPSHOT_SECTIONS)
    if extra:
        errors.append(f"unknown sections: {sorted(extra)}")
    if errors:
        return VerifyReport(False, errors, version)

    if doc.get("checksum") != checksum(data):
        errors.append("checksum mismatch (file is corrupted or edited)")
    counts = {s: len(data[s]) for s in SNAPSHOT_SECTIONS}
    if doc.get("counts") != counts:
        errors.append(f"counts mismatch: header {doc.get('counts')} vs data {counts}")

    records = _parse_records(data, errors)
    if records is not None:
        _check_integrity(records, data["settings"], errors)

    if not errors:
        # Trial import into a throwaway in-memory store to catch DB-level constraints.
        trial = SQLiteStore(":memory:")
        try:
            trial.import_snapshot(data)
        except StoreError as exc:
            errors.append(f"trial import failed: {exc}")
        finally:
            trial.close()
    return VerifyReport(not errors, errors, version, counts)


def _parse_records(data: dict[str, Any], errors: list[str]) -> dict[str, list[Any]] | None:
    records: dict[str, list[Any]] = {}
    start = len(errors)
    for section, cls in _RECORD_TYPES.items():
        records[section] = []
        for i, raw in enumerate(data[section]):
            where = f"{section}[{i}]"
            if not isinstance(raw, dict):
                errors.append(f"{where}: not an object")
                continue
            try:
                rec = cls.from_dict(raw)
            except (ValueError, TypeError) as exc:
                errors.append(f"{where}: {exc}")
                continue
            if not is_uuid(rec.id):
                errors.append(f"{where}: id is not a UUID: {rec.id!r}")
            for field in _TIMESTAMP_FIELDS.get(section, ()):
                value = getattr(rec, field)
                if value is None and field in ("finished_at", "occurred_at", "approved_at"):
                    continue
                try:
                    parse_iso(value)
                except (TypeError, ValueError):
                    errors.append(f"{where}.{field}: invalid or naive timestamp {value!r}")
            for field in _REQUIRED_TEXT.get(section, ()):
                value = getattr(rec, field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{where}.{field}: required text is empty")
            records[section].append(rec)
    for i, raw in enumerate(data["settings"]):
        if not isinstance(raw, dict) or set(raw) != {"key", "value"} or not isinstance(raw.get("key"), str):
            errors.append(f"settings[{i}]: needs exactly key and value")
        elif _looks_secret(raw["key"], raw["value"]):
            errors.append(f"settings[{i}]: secret-looking setting {raw['key']!r} is not allowed")
    return records if len(errors) == start else None


def _check_integrity(r: dict[str, list[Any]], settings: list[dict], errors: list[str]) -> None:
    for section, recs in r.items():
        dupes = [k for k, n in Counter(x.id for x in recs).items() if n > 1]
        if dupes:
            errors.append(f"{section}: duplicate ids {dupes[:3]}")
    dup_keys = [k for k, n in Counter(s["key"] for s in settings).items() if n > 1]
    if dup_keys:
        errors.append(f"settings: duplicate keys {dup_keys}")

    sessions = {s.id for s in r["sessions"]}
    personas = {p.id: p for p in r["persona_versions"]}
    turns = {t.id: t for t in r["turns"]}
    messages = {m.id: m for m in r["messages"]}
    excerpts = {e.id for e in r["source_excerpts"]}
    versions = {v.id: v for v in r["memory_versions"]}
    memory_ids = {v.memory_id for v in r["memory_versions"]}
    entities = {e.id for e in r["entities"]}

    active_keys = Counter(p.persona_key for p in personas.values() if p.status == PersonaStatus.ACTIVE)
    for key, n in active_keys.items():
        if n > 1:
            errors.append(f"persona {key}: {n} active versions")

    for t in turns.values():
        if t.session_id not in sessions:
            errors.append(f"turn {t.id}: unknown session {t.session_id}")
        for pid in t.persona_version_ids:
            if pid not in personas:
                errors.append(f"turn {t.id}: unknown persona version {pid}")
        for vid in t.context_memory_version_ids:
            if vid not in versions:
                errors.append(f"turn {t.id}: unknown memory version {vid}")
        for mid in t.context_message_ids:
            if mid not in messages:
                errors.append(f"turn {t.id}: unknown context message {mid}")
    for key, n in Counter((t.session_id, t.seq) for t in turns.values()).items():
        if n > 1:
            errors.append(f"turns: duplicate seq {key}")

    for m in messages.values():
        if m.session_id not in sessions:
            errors.append(f"message {m.id}: unknown session {m.session_id}")
        turn = turns.get(m.turn_id)
        if turn is None:
            errors.append(f"message {m.id}: unknown turn {m.turn_id}")
        elif turn.session_id != m.session_id:
            errors.append(f"message {m.id}: turn belongs to another session")
    for key, n in Counter((m.session_id, m.seq) for m in messages.values()).items():
        if n > 1:
            errors.append(f"messages: duplicate seq {key}")

    successors: dict[str, list[MemoryVersion]] = {}
    approved_per_memory: Counter = Counter()
    for v in versions.values():
        try:
            validate_memory_fields(v.body, v.kind, v.perspective, v.occurred_at)
        except ValueError as exc:
            errors.append(f"memory version {v.id}: {exc}")
        for mid in v.source_message_ids:
            if mid not in messages:
                errors.append(f"memory version {v.id}: unknown source message {mid}")
        for eid in v.source_excerpt_ids:
            if eid not in excerpts:
                errors.append(f"memory version {v.id}: unknown source excerpt {eid}")
        if v.status == MemoryStatus.APPROVED:
            approved_per_memory[v.memory_id] += 1
        if v.supersedes_version_id is not None:
            prev = versions.get(v.supersedes_version_id)
            if prev is None:
                errors.append(f"memory version {v.id}: supersedes unknown version {v.supersedes_version_id}")
            else:
                if prev.memory_id != v.memory_id:
                    errors.append(f"memory version {v.id}: supersedes a version of another memory")
                if prev.status != MemoryStatus.SUPERSEDED:
                    errors.append(f"memory version {prev.id}: replaced but status is {prev.status.value}")
                successors.setdefault(prev.id, []).append(v)
    for v in versions.values():
        if v.status == MemoryStatus.SUPERSEDED and not successors.get(v.id):
            errors.append(f"memory version {v.id}: superseded without a successor")
        if len(successors.get(v.id, [])) > 1:
            errors.append(f"memory version {v.id}: replaced by more than one version")
    for mem, n in approved_per_memory.items():
        if n > 1:
            errors.append(f"memory {mem}: {n} approved versions")

    def node_exists(node_type: NodeType, node_id: str) -> bool:
        if node_type == NodeType.ENTITY:
            return node_id in entities
        if node_type == NodeType.MEMORY_VERSION:
            return node_id in versions
        return node_id in memory_ids

    for link in r["links"]:
        if not node_exists(link.src_type, link.src_id):
            errors.append(f"link {link.id}: unknown {link.src_type.value} {link.src_id}")
        if not node_exists(link.dst_type, link.dst_id):
            errors.append(f"link {link.id}: unknown {link.dst_type.value} {link.dst_id}")


def load_document(path: str | Path) -> tuple[Any, list[str]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), []
    except FileNotFoundError:
        return None, [f"file not found: {path}"]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, [f"file is not valid JSON: {exc}"]


def verify_file(path: str | Path) -> VerifyReport:
    doc, errors = load_document(path)
    if errors:
        return VerifyReport(False, errors)
    return verify_document(doc)


# ------------------------------------------------------------------------ import


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _remove_db_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def import_file(
    path: str | Path,
    target_db_path: str | Path,
    active_db_path: str | Path | None = None,
    timezone: str | None = None,
) -> ImportResult:
    """Validate `path` and import it into `target_db_path` (never the active DB).

    A new target is built in a temp file and renamed only after the import and a
    re-export comparison succeed. An existing target receives a single all-or-nothing
    transaction (same id + same content is skipped, different content aborts).
    """
    target = Path(target_db_path)
    if active_db_path is not None and _same_file(target, Path(active_db_path)):
        raise ImportValidationError(["target is the database currently in use; choose another file"])
    doc, load_errors = load_document(path)
    if load_errors:
        raise ImportValidationError(load_errors)
    report = verify_document(doc)
    if not report.ok:
        raise ImportValidationError(report.errors)
    data = doc["data"]
    kwargs = {"timezone": timezone} if timezone else {}

    if target.exists():
        store = SQLiteStore(target, **kwargs)
        try:
            stats = store.import_snapshot(data)
        finally:
            store.close()
        return ImportResult(str(target), False, stats.inserted, stats.skipped)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.importing-{uuid.uuid4().hex}")
    try:
        store = SQLiteStore(tmp, **kwargs)
        try:
            stats = store.import_snapshot(data)
            if store.export_snapshot() != _normalised(data):
                raise ImportValidationError(["imported database does not match the export file"])
            store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            store.close()
        os.replace(tmp, target)
    finally:
        _remove_db_files(tmp)
    return ImportResult(str(target), True, stats.inserted, stats.skipped)


def _normalised(data: dict[str, Any]) -> dict[str, Any]:
    """Round-trip the input through a scratch store so ordering matches export_snapshot."""
    scratch = SQLiteStore(":memory:")
    try:
        scratch.import_snapshot(data)
        return scratch.export_snapshot()
    finally:
        scratch.close()
