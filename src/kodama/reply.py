"""モデル出力（design.md §6）の検証。

検証に通らない出力は会話として保存しない。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from kodama.gestures import capabilities, normalize_gesture

KINDS = ("user_stated", "character_view", "imagination")
MAX_UTTERANCES = 4
MAX_TEXT_CHARS = 1000
MAX_CANDIDATES = 5


def reply_schema(character_ids: Sequence[str], gesture_capabilities=None) -> dict[str, Any]:
    """出力の JSON Schema。話者の enum は人物設定パックのキャラクターID。"""
    speakers = list(character_ids)
    allowed = capabilities(speakers, gesture_capabilities)
    return {
        "type": "object",
        "properties": {
            "utterances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": speakers},
                        "text": {"type": "string"},
                        "gesture": {"type": "string", "enum": list(dict.fromkeys(["none", *(g for ids in allowed.values() for g in ids)]))},
                    },
                    "required": ["speaker", "text"],
                    "additionalProperties": False,
                },
            },
            "memory_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "body": {"type": "string"},
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "perspective": {"anyOf": [{"type": "string", "enum": speakers}, {"type": "null"}]},
                        "subjects": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["body", "kind", "perspective", "subjects", "tags"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["utterances", "memory_candidates"],
        "additionalProperties": False,
    }


class InvalidReply(Exception):
    pass


@dataclass(frozen=True)
class ParsedReply:
    utterances: list[tuple[str, str]]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gestures: list[str] = field(default_factory=list)


_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n(.*)\n```\Z", re.DOTALL)


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidReply(f"JSONとして読めない出力です: {e.msg}") from None


def _str_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    return [x.strip() for x in value if x.strip()]


def _parse_candidate(c: Any, speakers: Sequence[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(c, dict):
        return None, "記憶候補がオブジェクトではありません"
    body = c.get("body")
    if not isinstance(body, str) or not body.strip():
        return None, "記憶候補の本文が空です"
    kind = c.get("kind")
    if kind not in KINDS:
        return None, f"記憶候補の種類が不正です: {kind!r}"
    perspective = c.get("perspective")
    if kind == "character_view":
        if perspective not in speakers:
            return None, f"受け取り方の候補に視点（{'/'.join(speakers)}）がありません"
    elif perspective is not None:
        return None, "視点は受け取り方の候補にだけ付けられます"
    subjects = _str_list(c.get("subjects"))
    tags = _str_list(c.get("tags"))
    if subjects is None or tags is None:
        return None, "記憶候補の対象・タグが文字列の配列ではありません"
    return {
        "body": body.strip(),
        "kind": kind,
        "perspective": perspective,
        "subjects": subjects,
        "tags": tags,
    }, None


def parse_reply(raw: str, character_ids: Sequence[str], gesture_capabilities=None) -> ParsedReply:
    speakers = tuple(character_ids)
    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise InvalidReply("出力がJSONオブジェクトではありません")
    unknown = set(data) - {"utterances", "memory_candidates"}
    if unknown:
        raise InvalidReply(f"未知の項目があります: {sorted(unknown)}")
    utts = data.get("utterances")
    if not isinstance(utts, list) or not (1 <= len(utts) <= MAX_UTTERANCES):
        raise InvalidReply(f"発話は1〜{MAX_UTTERANCES}件の配列である必要があります")
    utterances: list[tuple[str, str]] = []
    gestures: list[str] = []
    for u in utts:
        if not isinstance(u, dict) or set(u) - {"speaker", "text", "gesture"}:
            raise InvalidReply("発話の形式が不正です")
        speaker = u.get("speaker")
        if speaker not in speakers:
            raise InvalidReply(f"不明な話者です: {speaker!r}")
        text = u.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidReply("空の発話があります")
        text = text.strip()
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidReply(f"発話が長すぎます（{len(text)}字）")
        utterances.append((speaker, text))
        gestures.append(normalize_gesture(speaker, u.get("gesture"), gesture_capabilities))

    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    raw_cands = data.get("memory_candidates", [])
    if raw_cands is None:
        raw_cands = []
    if not isinstance(raw_cands, list):
        warnings.append("記憶候補が配列ではないため捨てました")
        raw_cands = []
    for c in raw_cands:
        if len(candidates) >= MAX_CANDIDATES:
            warnings.append(f"記憶候補が{MAX_CANDIDATES}件を超えたため残りを捨てました")
            break
        parsed, warn = _parse_candidate(c, speakers)
        if parsed is None:
            warnings.append(f"不正な記憶候補を捨てました: {warn}")
        else:
            candidates.append(parsed)
    return ParsedReply(utterances=utterances, candidates=candidates, warnings=warnings, gestures=gestures)
