"""1ターンの処理: 入力保存 → モデル呼び出し（トランザクション外・1回）→ 検証 → 返答保存＋完了。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from kodama.config import Config
from kodama.context import ContextPlan, build_context, to_model_request
from kodama.domain import MemoryDraft, MemoryKind, MemoryOrigin, MemoryStatus, MemoryVersion, Message, PersonaVersion, Speaker, Turn, TurnStatus, Usage
from kodama.model.base import (
    ModelAdapter,
    ModelAPIError,
    ModelConfigError,
    ModelConnectionError,
    ModelError,
    ModelRefusal,
    ModelTimeout,
    redact,
)
from kodama.personas import PERSONA_KEYS, PersonaFile, load_persona_files, unified_diff
from kodama.reply import InvalidReply, parse_reply


@dataclass
class TurnOutcome:
    ok: bool
    turn_id: str | None
    user_message_id: str | None = None
    replies: list[Message] = field(default_factory=list)
    error_kind: str | None = None  # invalid_reply / timeout / connection / api / refusal / config / interrupted / storage / unexpected
    error_message: str | None = None
    outcome_unknown: bool = False  # 送信後に結果が分からない（課金の有無も不明）
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    is_mock: bool = False
    warnings: list[str] = field(default_factory=list)
    plan: ContextPlan | None = None
    saved_memories: list[MemoryVersion] = field(default_factory=list)  # このターンで保存した記憶（自動承認分）


@dataclass
class PersonaReport:
    registered: list[str] = field(default_factory=list)  # 初回登録したキー
    pending: dict[str, tuple[PersonaVersion, PersonaFile]] = field(default_factory=dict)  # 未承認の変更


def sync_personas(store, personas_dir: str | Path) -> PersonaReport:
    """ファイルと有効版を比べる。有効版がなければ初回登録し、差があれば旧版のまま「未承認」として返す。"""
    report = PersonaReport()
    files = load_persona_files(personas_dir)
    for key in PERSONA_KEYS:
        f = files[key]
        active = store.get_active_persona(key)
        if active is None:
            store.activate_persona_version(key, f.body, f.path, note="初回登録")
            report.registered.append(key)
        elif active.body != f.body:
            report.pending[key] = (active, f)
    return report


def persona_diff(active: PersonaVersion, f: PersonaFile) -> str:
    return unified_diff(active.body, f.body, f.key)


def approve_persona(store, personas_dir: str | Path, key: str) -> PersonaVersion:
    f = load_persona_files(personas_dir)[key]
    return store.activate_persona_version(key, f.body, f.path, note="本人が差分を確認して承認")


_ERROR_KINDS: list[tuple[type, str]] = [
    (ModelTimeout, "timeout"),
    (ModelConnectionError, "connection"),
    (ModelRefusal, "refusal"),
    (ModelConfigError, "config"),
    (ModelAPIError, "api"),
]


class ConversationService:
    def __init__(self, store, adapter: ModelAdapter, config: Config, clock: Callable[[], datetime] | None = None):
        self.store = store
        self.adapter = adapter
        self.config = config
        self._tz = ZoneInfo(config.timezone)
        self._clock = clock or (lambda: datetime.now(self._tz))

    def startup(self) -> tuple[list[Turn], PersonaReport]:
        """未完了ターンを中断扱いにし（再送しない）、人物設定ファイルを確認する。"""
        recovered = self.store.recover_incomplete_turns()
        report = sync_personas(self.store, self.config.personas_dir)
        return recovered, report

    def preview(self, session_id: str, text: str) -> ContextPlan:
        return build_context(self.store, self.config, session_id, text, self._clock())

    def send(self, session_id: str, text: str) -> TurnOutcome:
        plan = self.preview(session_id, text)
        request = to_model_request(plan, self.config)
        turn, user_msg = self.store.begin_turn(
            session_id,
            text,
            self.adapter.provider,
            self.adapter.model,
            plan.persona_version_ids,
            plan.memory_version_ids,
            plan.message_ids,
        )
        outcome = TurnOutcome(ok=False, turn_id=turn.id, user_message_id=user_msg.id, is_mock=bool(self.adapter.is_mock), plan=plan)

        try:
            result = self.adapter.generate(request)
        except KeyboardInterrupt:
            outcome.error_kind = "interrupted"
            outcome.outcome_unknown = not self.adapter.is_mock
            outcome.error_message = "中断しました。" + ("" if self.adapter.is_mock else "送信済みかどうか・課金の有無は不明です。")
            self.store.fail_turn(turn.id, TurnStatus.INTERRUPTED, "利用者が中断")
            return outcome
        except ModelError as e:
            kind = next((k for cls, k in _ERROR_KINDS if isinstance(e, cls)), "api")
            outcome.error_kind = kind
            outcome.error_message = redact(str(e))
            outcome.outcome_unknown = e.outcome_unknown
            outcome.input_tokens, outcome.output_tokens = e.input_tokens, e.output_tokens
            outcome.cache_read_tokens, outcome.cache_write_tokens = e.cache_read_tokens, e.cache_write_tokens
            self.store.fail_turn(turn.id, TurnStatus.FAILED, f"{kind}: {outcome.error_message}",
                                 Usage(e.input_tokens, e.output_tokens, e.cache_read_tokens, e.cache_write_tokens))
            return outcome
        except Exception as e:  # 想定外。内容は秘密を伏せて要約だけ残す
            outcome.error_kind = "unexpected"
            outcome.error_message = redact(f"{type(e).__name__}: {e}")
            outcome.outcome_unknown = not self.adapter.is_mock
            self.store.fail_turn(turn.id, TurnStatus.FAILED, f"unexpected: {outcome.error_message}")
            return outcome

        outcome.input_tokens, outcome.output_tokens = result.input_tokens, result.output_tokens
        outcome.is_mock = bool(result.is_mock)
        outcome.cache_read_tokens, outcome.cache_write_tokens = result.cache_read_tokens, result.cache_write_tokens
        usage = Usage(result.input_tokens, result.output_tokens, result.cache_read_tokens, result.cache_write_tokens)
        try:
            parsed = parse_reply(result.raw_text)
        except InvalidReply as e:
            outcome.error_kind = "invalid_reply"
            outcome.error_message = f"モデルの返答が形式に合わないため、会話として保存しませんでした（{e}）"
            self.store.fail_turn(turn.id, TurnStatus.FAILED, f"invalid_reply: {e}", usage)
            return outcome

        drafts: list[MemoryDraft] = []
        warnings = list(parsed.warnings)
        for c in parsed.candidates:
            try:
                draft = MemoryDraft(
                    body=c["body"],
                    kind=MemoryKind(c["kind"]),
                    perspective=Speaker(c["perspective"]) if c["perspective"] else None,
                    subjects=tuple(c["subjects"]),
                    tags=tuple(c["tags"]),
                    source_message_ids=(user_msg.id,),
                )
                draft.validate()
                drafts.append(draft)
            except ValueError as e:
                warnings.append(f"不正な記憶候補を捨てました: {e}")
        outcome.warnings = warnings

        try:
            auto = self.config.memory_auto_approve
            replies = self.store.complete_turn(
                turn.id, parsed.utterances, usage, drafts, model=result.model,
                candidate_status=MemoryStatus.APPROVED if auto else MemoryStatus.CANDIDATE,
            )
        except Exception as e:
            outcome.error_kind = "storage"
            outcome.error_message = redact(f"返答を保存できませんでした: {type(e).__name__}: {e}")
            self.store.fail_turn(turn.id, TurnStatus.FAILED, outcome.error_message, usage)
            return outcome
        outcome.ok = True
        outcome.replies = replies
        if self.config.memory_auto_approve and drafts:
            try:
                outcome.saved_memories = [
                    v for v in self.store.list_memory_versions([MemoryStatus.APPROVED])
                    if user_msg.id in v.source_message_ids and v.origin == MemoryOrigin.MODEL_CANDIDATE
                ]
            except Exception:
                pass
        return outcome
