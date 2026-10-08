"""対話CLI。台詞は 蓮「…」/葵「…」、操作案内やエラーは「操作:」で会話と区別する。"""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from kodama import migration
from kodama.config import Config, ConfigError, load_config
from kodama.context import SPEAKER_LABEL, PersonaMissing
from kodama.conversation import ConversationService, PersonaReport, approve_persona, persona_diff, sync_personas
from kodama.domain import (
    EntityKind,
    MemoryDraft,
    MemoryKind,
    MemoryOrigin,
    MemoryStatus,
    MemoryVersion,
    NodeType,
    Session,
    Speaker,
    TurnStatus,
    parse_iso,
)
from kodama.model.base import ModelAdapter, redact
from kodama.model.mock import MockAdapter
from kodama.personas import PERSONA_KEYS
from kodama.recall import REASON_LABELS
from kodama.storage.base import ImportValidationError, StoreError
from kodama.storage.sqlite import SQLiteStore

HELP = """操作: 使えるコマンド
  /help                      この一覧
  /quit                      終了（Ctrl+D / Ctrl+C でも終了）
  /history                   現在のセッションの原文
  /session new [題名]        新しいセッションを始める
  /session list              セッション一覧
  /session resume <ID先頭>   セッションを再開する
  /memory list [all]         記憶と候補の一覧（all で却下・訂正済み・無効化も）
  /memory show <ID先頭>      記憶の詳細（出典・経緯）
  /memory add <本文> [--kind user_stated|character_view|imagination] [--perspective ren|aoi]
              [--tags a,b] [--about 名前] [--excerpt <抜粋ID先頭>]
                             記憶を登録（出典は直前のあなたの発言、または --excerpt の抜粋）
  /memory recent [N]         会話から自動で残った記憶を新しい順に（既定10件）
  /memory approve|reject <ID先頭>   候補を承認・却下（自動で残った記憶への reject は無効化＝以後使わない）
  /memory revise <ID先頭> <新しい本文>  訂正（旧版は残り、通常の想起では使われない）
  /memory invalidate <ID先頭>       承認済みの記憶を無効化
  /excerpt add <題名> | <箇所> | <本文>   本人が選んだ資料の抜粋を出典として登録
  /excerpt list
  /context [仮の入力]        次の送信に含める内容を、送らずに表示
  /persona status|diff [名前]|approve <名前>|history [名前]   人物設定の版と変更の確認
  /export <ファイル>         全データを移行ファイルへ書き出す
  /import verify <ファイル>  移行ファイルを検証だけする
  /import <ファイル> <新しいDB>  検証して別のDBへ取り込む（使用中のDBには取り込まない）"""

STATUS_LABEL = {
    MemoryStatus.CANDIDATE: "承認待ち",
    MemoryStatus.APPROVED: "承認済み",
    MemoryStatus.REJECTED: "却下",
    MemoryStatus.SUPERSEDED: "訂正済み(旧版)",
    MemoryStatus.INVALIDATED: "無効",
}
KIND_SHORT = {
    MemoryKind.USER_STATED: "本人の話",
    MemoryKind.CHARACTER_VIEW: "受け取り方",
    MemoryKind.IMAGINATION: "想像",
}
ORIGIN_LABEL = {MemoryOrigin.USER_EXPLICIT: "本人登録", MemoryOrigin.MODEL_CANDIDATE: "会話から自動"}
_PERSON_NAMES = {"あなた", "蓮", "葵", "同居人"}


class Quit(Exception):
    pass


class App:
    def __init__(
        self,
        config: Config,
        store,
        adapter: ModelAdapter,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self.config = config
        self.store = store
        self.adapter = adapter
        self.input = input_fn
        self.out = output_fn
        self.tz = ZoneInfo(config.timezone)
        self.service = ConversationService(store, adapter, config)
        self.session: Session | None = None

    # ------------------------------------------------------------------ output

    def op(self, text: str) -> None:
        for i, line in enumerate(redact(text).splitlines() or [""]):
            self.out(("操作: " if i == 0 else "      ") + line)

    def say(self, speaker: Speaker, text: str) -> None:
        self.out(f"{SPEAKER_LABEL[Speaker(speaker)]}「{text}」")

    def fmt_time(self, iso: str) -> str:
        return parse_iso(iso).astimezone(self.tz).strftime("%m/%d %H:%M")

    # ----------------------------------------------------------------- startup

    def start(self, new_session: bool = False) -> None:
        recovered, report = self.service.startup()
        if self.adapter.is_mock:
            self.op("モックモードです。返答は固定のモック応答で、実際のLLMの応答ではありません。")
        else:
            self.op(f"実APIモード: {self.adapter.provider} / {self.adapter.model}")
        if recovered:
            self.op(f"前回、応答を待ったまま終了したターンが {len(recovered)} 件ありました。中断として記録し、再送はしていません。")
        self._report_personas(report)
        sessions = self.store.list_sessions()
        if new_session or not sessions:
            self.session = self.store.create_session(self._default_title())
            self.op(f"新しいセッションを始めました: {self.session.title} ({self.session.id[:8]})")
        else:
            self.session = sessions[-1]
            self.op(f"前回のセッションを続けます: {self.session.title} ({self.session.id[:8]})  新しく始めるには /session new")
            self._show_tail(6)
        self.op("/help で操作の一覧。")

    def _default_title(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m-%d %H:%M")

    def _report_personas(self, report: PersonaReport) -> None:
        if report.registered:
            self.op(f"人物設定を初回登録しました: {', '.join(report.registered)}")
        if report.pending:
            keys = ", ".join(report.pending)
            self.op(
                f"人物設定ファイルに未承認の変更があります: {keys}\n"
                "承認するまでは以前の版を使います。/persona diff で差分、/persona approve <名前> で反映。"
            )

    def _show_tail(self, n: int) -> None:
        entries = self.store.recent_messages(self.session.id, n)
        if entries:
            self.op("直前の会話:")
            for e in entries:
                self._print_message(e.message, e.turn_status)

    def _print_message(self, m, status: TurnStatus | None = None) -> None:
        mark = ""
        if m.speaker == Speaker.USER and status in (TurnStatus.FAILED, TurnStatus.INTERRUPTED):
            mark = f"  ［未応答: {'失敗' if status == TurnStatus.FAILED else '中断'}］"
        self.out(f"[{self.fmt_time(m.created_at)}] {SPEAKER_LABEL[m.speaker]}「{m.text}」{mark}")

    # -------------------------------------------------------------------- loop

    def run(self, new_session: bool = False) -> int:
        self.start(new_session)
        while True:
            try:
                line = self.input("あなた> ")
            except (EOFError, KeyboardInterrupt):
                self.out("")
                self.op("終了します。")
                return 0
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("/"):
                    self.command(line)
                else:
                    self.talk(line)
            except Quit:
                self.op("終了します。")
                return 0
            except KeyboardInterrupt:
                self.op("中断しました。")
            except (StoreError, ValueError, PersonaMissing, ImportValidationError) as e:
                self.op(f"エラー: {e}")
            except Exception as e:  # 想定外のエラーも台詞にせず、秘密を伏せて表示
                self.op(f"予期しないエラー: {type(e).__name__}: {e}")

    def talk(self, text: str) -> None:
        outcome = self.service.send(self.session.id, text)
        if not outcome.ok:
            msg = {
                "timeout": "応答がタイムアウトしました。",
                "connection": "接続できませんでした。",
                "refusal": "モデルが応答を辞退しました。",
                "config": "設定の問題で送信できませんでした。",
                "api": "APIがエラーを返しました。",
                "invalid_reply": "モデルの返答が形式に合いませんでした。",
                "interrupted": "中断しました。",
                "storage": "返答を保存できませんでした。",
                "unexpected": "予期しないエラーが起きました。",
            }.get(outcome.error_kind, "失敗しました。")
            detail = outcome.error_message or ""
            lines = [msg, "" if detail.strip() == msg else detail]
            if outcome.outcome_unknown:
                lines.append("送信済みかどうか・課金の有無は不明です。")
            lines.append("入力は記録済みです。自動では再送しません。必要ならもう一度入力してください。")
            self.op("\n".join(x for x in lines if x))
            self._usage_line(outcome)
            return
        if outcome.is_mock:
            self.out("[モック応答]")
        for m in outcome.replies:
            self.say(m.speaker, m.text)
        for w in outcome.warnings:
            self.op(w)
        if self.config.show_memory_notices and outcome.saved_memories:
            self.op("記憶に残しました: " + " / ".join(
                f"{v.id[:8]} {v.body[:20]}{'…' if len(v.body) > 20 else ''}" for v in outcome.saved_memories))
        new_cands = [v for v in self.store.list_memory_versions([MemoryStatus.CANDIDATE])
                     if outcome.user_message_id in v.source_message_ids]
        if new_cands:
            self.op(f"記憶の候補が {len(new_cands)} 件あります（未承認）。/memory list で確認できます。")
        self._usage_line(outcome)

    def _usage_line(self, outcome) -> None:
        if outcome.is_mock:
            return
        o = "不明" if outcome.output_tokens is None else str(outcome.output_tokens)
        parts = [outcome.input_tokens, outcome.cache_read_tokens, outcome.cache_write_tokens]
        if outcome.input_tokens is None:
            i = "不明"
        else:
            i = str(sum(p or 0 for p in parts))  # 合計入力 = 通常入力 + キャッシュ読込 + キャッシュ書込
        detail = []
        if outcome.cache_read_tokens is not None:
            detail.append(f"キャッシュ読込 {outcome.cache_read_tokens}")
        if outcome.cache_write_tokens is not None:
            detail.append(f"書込 {outcome.cache_write_tokens}")
        if detail:
            i += f"（うち{' / '.join(detail)}）"
        self.op(f"利用量: 入力 {i}・出力 {o} トークン（料金は表示しません）")

    # ---------------------------------------------------------------- commands

    def command(self, line: str) -> None:
        try:
            args = shlex.split(line)
        except ValueError as e:
            self.op(f"コマンドを読めません: {e}")
            return
        name, rest = args[0], args[1:]
        handler = {
            "/help": lambda r: self.op(HELP.removeprefix("操作: ")),
            "/quit": self._quit,
            "/exit": self._quit,
            "/history": self.cmd_history,
            "/session": self.cmd_session,
            "/memory": self.cmd_memory,
            "/excerpt": lambda r: self.cmd_excerpt(line),
            "/context": lambda r: self.cmd_context(line),
            "/persona": self.cmd_persona,
            "/export": self.cmd_export,
            "/import": self.cmd_import,
        }.get(name)
        if handler is None:
            self.op(f"不明なコマンドです: {name}（/help で一覧）")
            return
        handler(rest)

    def _quit(self, rest) -> None:
        raise Quit()

    def cmd_history(self, rest) -> None:
        turns = {t.id: t.status for t in self.store.list_turns(self.session.id)}
        msgs = self.store.list_messages(self.session.id)
        self.op(f"セッション {self.session.title} ({self.session.id[:8]}) の原文 {len(msgs)} 件")
        for m in msgs:
            self._print_message(m, turns.get(m.turn_id))

    def cmd_session(self, rest) -> None:
        sub = rest[0] if rest else "list"
        if sub == "new":
            title = " ".join(rest[1:]) or self._default_title()
            self.session = self.store.create_session(title)
            self.op(f"新しいセッションを始めました: {title} ({self.session.id[:8]})")
        elif sub == "list":
            for s in self.store.list_sessions():
                n = len(self.store.list_messages(s.id))
                cur = " ←現在" if self.session and s.id == self.session.id else ""
                self.op(f"{s.id[:8]}  {self.fmt_time(s.created_at)}  {s.title}  ({n}件){cur}")
        elif sub == "resume" and len(rest) >= 2:
            matches = [s for s in self.store.list_sessions() if s.id.startswith(rest[1])]
            if len(matches) != 1:
                self.op("該当するセッションが一つに決まりません。/session list でIDを確認してください。")
                return
            self.session = matches[0]
            self.op(f"セッションを再開しました: {self.session.title} ({self.session.id[:8]})")
            self._show_tail(6)
        else:
            self.op("使い方: /session new [題名] | list | resume <ID先頭>")

    # memory ------------------------------------------------------------------

    def _find_version(self, prefix: str) -> MemoryVersion:
        matches = [v for v in self.store.list_memory_versions() if v.id.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"記憶ID {prefix!r} が一つに決まりません（{len(matches)}件）")
        return matches[0]

    def _memory_line(self, v: MemoryVersion) -> str:
        view = f"・{SPEAKER_LABEL[v.perspective]}" if v.perspective else ""
        tags = f"  #{' #'.join(v.tags)}" if v.tags else ""
        src = "" if v.has_source else "  (出典なし)"
        return f"{v.id[:8]} [{STATUS_LABEL[v.status]}・{ORIGIN_LABEL[v.origin]}] {KIND_SHORT[v.kind]}{view}: {v.body}{tags}{src}"

    def cmd_memory(self, rest) -> None:
        sub = rest[0] if rest else "list"
        if sub == "list":
            statuses = None if rest[1:2] == ["all"] else [MemoryStatus.CANDIDATE, MemoryStatus.APPROVED]
            versions = self.store.list_memory_versions(statuses)
            if not versions:
                self.op("記憶はまだありません。")
            for v in versions:
                self.op(self._memory_line(v))
        elif sub == "recent":
            try:
                n = int(rest[1]) if len(rest) > 1 else 10
                if n <= 0:
                    raise ValueError
            except ValueError:
                self.op("件数は正の整数で指定してください。")
                return
            recent = [v for v in self.store.list_memory_versions()
                      if v.origin == MemoryOrigin.MODEL_CANDIDATE]
            recent.sort(key=lambda v: (v.recorded_at, v.id), reverse=True)
            if not recent:
                self.op("会話から自動で残った記憶はまだありません。")
            for v in recent[:n]:
                self.op(f"{self.fmt_time(v.recorded_at)} {self._memory_line(v)}")
        elif sub == "show" and len(rest) >= 2:
            self._memory_show(self._find_version(rest[1]))
        elif sub == "add" and len(rest) >= 2:
            self._memory_add(rest[1:])
        elif sub in ("approve", "reject", "invalidate") and len(rest) >= 2:
            v = self._find_version(rest[1])
            new = {"approve": MemoryStatus.APPROVED, "reject": MemoryStatus.REJECTED,
                   "invalidate": MemoryStatus.INVALIDATED}[sub]
            reason = "本人の操作"
            if sub == "reject" and v.status == MemoryStatus.APPROVED:
                # 自動承認済みの記憶の却下は、既存の状態遷移どおり approved -> invalidated に読み替える
                new, reason = MemoryStatus.INVALIDATED, "本人が取り消し（自動で残った記憶の却下）"
            updated = self.store.set_memory_status(v.id, new, reason)
            self.op(self._memory_line(updated))
        elif sub == "revise" and len(rest) >= 3:
            v = self._find_version(rest[1])
            body = " ".join(rest[2:])
            draft = MemoryDraft(body=body, kind=v.kind, perspective=v.perspective, subjects=v.subjects,
                                tags=v.tags, aliases=v.aliases, occurred_at=v.occurred_at,
                                source_message_ids=v.source_message_ids, source_excerpt_ids=v.source_excerpt_ids)
            new = self.store.revise_memory(v.id, draft, "本人の訂正")
            self.op(f"訂正しました。新しい版: {self._memory_line(new)}\n旧版 {v.id[:8]} は訂正済みとして残ります。")
        else:
            self.op("使い方は /help を見てください。")

    def _memory_show(self, v: MemoryVersion) -> None:
        self.op(self._memory_line(v))
        self.op(f"ID: {v.id}\n論理記憶ID: {v.memory_id}\n記録: {self.fmt_time(v.recorded_at)}"
                f"  出来事: {self.fmt_time(v.occurred_at) if v.occurred_at else '不明'}"
                f"\n由来: {'本人の登録' if v.origin == MemoryOrigin.USER_EXPLICIT else 'モデルの候補'}"
                f"  対象: {', '.join(v.subjects) or '-'}  別名: {', '.join(v.aliases) or '-'}")
        for mid in v.source_message_ids:
            m = self.store.get_message(mid)
            self.op(f"出典(発言) {mid[:8]}: [{self.fmt_time(m.created_at)}] {SPEAKER_LABEL[m.speaker]}「{m.text}」")
        for eid in v.source_excerpt_ids:
            ex = self.store.get_source_excerpt(eid)
            self.op(f"出典(抜粋) {eid[:8]}: {ex.title} / {ex.locator}: {ex.text}")
        history = self.store.list_memory_versions(memory_id=v.memory_id)
        if len(history) > 1:
            self.op("経緯:")
            for h in sorted(history, key=lambda h: h.recorded_at):
                reason = f"（{h.status_reason}）" if h.status_reason else ""
                self.op(f"  {self.fmt_time(h.recorded_at)} {self._memory_line(h)}{reason}")

    def _memory_add(self, args: list[str]) -> None:
        p = argparse.ArgumentParser(prog="/memory add", add_help=False, exit_on_error=False)
        p.add_argument("body", nargs="+")
        p.add_argument("--kind", default="user_stated", choices=[k.value for k in MemoryKind])
        p.add_argument("--perspective", choices=["ren", "aoi"])
        p.add_argument("--tags", default="")
        p.add_argument("--about", action="append", default=[])
        p.add_argument("--excerpt")
        try:
            ns = p.parse_args(args)
        except (argparse.ArgumentError, SystemExit) as e:
            self.op(f"引数が読めません: {e}")
            return
        msg_ids: tuple[str, ...] = ()
        ex_ids: tuple[str, ...] = ()
        if ns.excerpt:
            found = [e for e in self.store.list_source_excerpts() if e.id.startswith(ns.excerpt)]
            if len(found) != 1:
                self.op("抜粋IDが一つに決まりません。/excerpt list で確認してください。")
                return
            ex_ids = (found[0].id,)
        else:
            mine = [m for m in self.store.list_messages(self.session.id) if m.speaker == Speaker.USER]
            if not mine:
                self.op("出典になるあなたの発言がありません。先に話すか、--excerpt で抜粋を指定してください。")
                return
            msg_ids = (mine[-1].id,)
        draft = MemoryDraft(
            body=" ".join(ns.body),
            kind=MemoryKind(ns.kind),
            perspective=Speaker(ns.perspective) if ns.perspective else None,
            tags=tuple(t for t in ns.tags.split(",") if t),
            subjects=tuple(ns.about),
            source_message_ids=msg_ids,
            source_excerpt_ids=ex_ids,
        )
        v = self.store.add_memory(draft, MemoryOrigin.USER_EXPLICIT, MemoryStatus.APPROVED)
        for name in ns.about:
            kind = EntityKind.PERSON if name in _PERSON_NAMES else EntityKind.TOPIC
            ent = self.store.upsert_entity(kind, name)
            self.store.add_link(NodeType.ENTITY, ent.id, NodeType.MEMORY, v.memory_id, "about")
        self.op(f"登録しました: {self._memory_line(v)}")

    def cmd_excerpt(self, line: str) -> None:
        body = line[len("/excerpt"):].strip()
        if body.startswith("add"):
            parts = [p.strip() for p in body[3:].split("|")]
            if len(parts) != 3 or not all(parts):
                self.op("使い方: /excerpt add <題名> | <箇所> | <本文>")
                return
            ex = self.store.add_source_excerpt(*parts)
            self.op(f"抜粋を登録しました: {ex.id[:8]} {ex.title} / {ex.locator}")
        else:
            for ex in self.store.list_source_excerpts():
                self.op(f"{ex.id[:8]} {ex.title} / {ex.locator}: {ex.text[:60]}")

    # context / persona -------------------------------------------------------

    def cmd_context(self, line: str) -> None:
        text = line[len("/context"):].strip() or "（仮の入力なし）"
        plan = self.service.preview(self.session.id, text)
        self.op("次の送信内容（送信はしていません）")
        self.op("送信先: " + ("なし（モック）" if self.adapter.is_mock else f"{self.adapter.provider} / {self.adapter.model}"))
        self.op("人物設定: " + ", ".join(f"{k}={plan.personas[k].id[:8]}" for k in PERSONA_KEYS))
        ids = [e.message.id[:8] for e in plan.recent]
        self.op(f"直近の会話: {len(ids)}件 " + (f"{ids[0]} 〜 {ids[-1]}" if ids else ""))
        if plan.dated_log:
            self.op(f"日付の会話記録 {plan.dated_log_label}: {len(plan.dated_log)}件")
        if plan.terms:
            self.op("検索語: " + "、".join(plan.terms))
        for r in plan.memories + plan.imaginations:
            v = r.version
            src = [f"発言{s[:8]}" for s in v.source_message_ids] + [f"抜粋{s[:8]}" for s in v.source_excerpt_ids]
            self.op(f"記憶 {v.id[:8]} ({KIND_SHORT[v.kind]}, 経路 {r.via}, 出典 {' '.join(src)}): {v.body}")
        for x in plan.excluded_memories:
            self.op(f"除外 {(x.version_id or x.memory_id)[:8]}: {REASON_LABELS.get(x.reason, x.reason)}")
        if plan.excluded_message_ids:
            self.op(f"量の上限で外した会話: {len(plan.excluded_message_ids)}件（原文は保存済み）")
        over = "（上限超過）" if plan.char_count > self.config.max_context_chars else ""
        self.op(f"文字数: {plan.char_count} / 上限 {self.config.max_context_chars}{over}")

    def cmd_persona(self, rest) -> None:
        sub = rest[0] if rest else "status"
        keys = [rest[1]] if len(rest) >= 2 else list(PERSONA_KEYS)
        if any(k not in PERSONA_KEYS for k in keys):
            self.op(f"人物設定の名前は {', '.join(PERSONA_KEYS)} です。")
            return
        report = sync_personas(self.store, self.config.personas_dir)
        if sub == "status":
            for k in PERSONA_KEYS:
                a = self.store.get_active_persona(k)
                state = "未承認の変更あり" if k in report.pending else "ファイルと一致"
                self.op(f"{k}: 有効版 {a.id[:8]}（{self.fmt_time(a.approved_at or a.created_at)}）{state}")
        elif sub == "diff":
            shown = False
            for k in keys:
                if k in report.pending:
                    self.out(persona_diff(*report.pending[k]))
                    shown = True
            if not shown:
                self.op("未承認の変更はありません。")
        elif sub == "approve" and len(rest) >= 2:
            if rest[1] not in report.pending:
                self.op("その人物設定に未承認の変更はありません。")
                return
            v = approve_persona(self.store, self.config.personas_dir, rest[1])
            self.op(f"{rest[1]} の新しい版 {v.id[:8]} を有効にしました。以前の版も本文ごと残っています。")
        elif sub == "history":
            for k in keys:
                for v in self.store.list_persona_versions(k):
                    self.op(f"{k} {v.id[:8]} [{v.status.value}] {self.fmt_time(v.created_at)} {v.note or ''}")
        else:
            self.op("使い方: /persona status | diff [名前] | approve <名前> | history [名前]")

    # migration ---------------------------------------------------------------

    def cmd_export(self, rest) -> None:
        if not rest:
            self.op("使い方: /export <ファイル>")
            return
        counts = migration.export_to_file(self.store, rest[0], settings=self.config.exportable())
        self.op(f"書き出しました: {rest[0]}\n" + ", ".join(f"{k} {n}" for k, n in counts.items()) +
                "\nこのファイルには会話の原文が含まれ、暗号化されていません。保管場所に注意してください。")

    def cmd_import(self, rest) -> None:
        if len(rest) == 2 and rest[0] == "verify":
            report = migration.verify_file(rest[1])
            if report.ok:
                self.op(f"検証OK（schema_version {report.schema_version}）: " +
                        ", ".join(f"{k} {n}" for k, n in report.counts.items()))
            else:
                self.op("検証NG:\n" + "\n".join(report.errors[:20]))
            return
        if len(rest) == 2:
            result = migration.import_file(rest[0], rest[1], active_db_path=self.config.db_path,
                                           timezone=self.config.timezone)
            self.op(f"{'新しいDBを作成して' if result.created_new else '既存のDBへ'}取り込みました: {result.target}\n"
                    f"追加: {_counts(result.inserted)}\n同一のため省略: {_counts(result.skipped)}\n"
                    f"使用中のDBは変わっていません。切り替えるには --db {result.target} で起動するか、設定の db_path を変更してください。")
            return
        self.op("使い方: /import verify <ファイル> | /import <ファイル> <新しいDB>")


# ---------------------------------------------------------------------- startup


def _counts(d: dict[str, int]) -> str:
    return ", ".join(f"{k} {n}" for k, n in d.items() if n) or "なし"


def make_adapter(config: Config) -> ModelAdapter:
    if config.provider == "claude":
        from kodama.model.claude import ClaudeAdapter

        return ClaudeAdapter(model=config.model, api_key_env=config.api_key_env, effort=config.effort)
    return MockAdapter()


def confirm_real_api(config: Config, input_fn, out) -> str:
    """実API送信の同意を取る。戻り値: 'claude' / 'mock' / 'quit'。"""
    out("操作: 実API（Anthropic Messages API）への送信が設定されています。")
    out(f"      送信先: Anthropic / モデル {config.model}（effort {config.effort}、出力上限 {config.max_tokens} トークン、"
        f"タイムアウト {config.timeout_seconds:g} 秒、自動再試行なし）")
    out(f"      送る内容: 人物設定3ファイル（{config.personas_dir}/）、このセッションの直近の会話（最大 {config.max_recent_messages} 件）、"
        f"「昨日」などと言ったときのその日の会話、想起した承認済みの記憶（最大 {config.max_memories} 件）、現在の入力")
    out("      送らないもの: APIキー以外の秘密、未承認の記憶候補、他のファイル。/context で送信前に中身を確認できます。")
    out("      API利用は課金の対象になり得ます。料金や契約による扱いはこのアプリでは判断しません。")
    try:
        ans = input_fn("実APIへの送信を有効にしますか？ [y/N] ").strip().lower()
        if ans in ("y", "yes"):
            return "claude"
        ans = input_fn("モックで続けますか？ [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    return "mock" if ans in ("y", "yes") else "quit"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kodama", description="蓮と葵と話すローカル会話CLI")
    p.add_argument("--config", help="設定ファイル（TOML）。省略時は ./kodama.toml があれば使う")
    p.add_argument("--db", help="使うDBファイル（設定の db_path を上書き）")
    p.add_argument("--provider", choices=["mock", "claude"], help="設定の provider を上書き")
    p.add_argument("--model", help="設定の model（モデルID）を上書き")
    p.add_argument("--new-session", action="store_true", help="新しいセッションで始める")
    return p


def main(argv: list[str] | None = None, input_fn=input, output_fn=print,
         adapter_factory: Callable[[Config], ModelAdapter] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config or ("kodama.toml" if Path("kodama.toml").exists() else None)
    try:
        config = load_config(config_path, {"db_path": args.db, "provider": args.provider, "model": args.model})
    except ConfigError as e:
        output_fn(f"操作: 設定エラー: {redact(str(e))}")
        return 2
    if config.provider == "claude":
        choice = confirm_real_api(config, input_fn, output_fn)
        if choice == "quit":
            output_fn("操作: 終了します。")
            return 0
        if choice == "mock":
            config = load_config(config_path, {"db_path": args.db, "provider": "mock", "model": args.model})
    Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(config.db_path, timezone=config.timezone)
    try:
        adapter = (adapter_factory or make_adapter)(config)
        app = App(config, store, adapter, input_fn, output_fn)
        return app.run(new_session=args.new_session)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
