"""人物の声の自動検査（必要条件のみ）。

ここで見るのは禁句・文末・形式などの機械的に判定できる条件だけ。合格しても
人物の声が再現できたことの証拠にはならない。自然さ・人物差は evals/README.md の手順で人が判断する。

検査の対象は人物設定パックの voice_cases.json で、話者ごとの規則は base_checks に話者IDで書く:
  "allowed_speakers": [...], "forbidden_phrases": {"<id>": [...], "any": [...]},
  "casual_endings_forbidden": ["<id>"],   # 常体の文末を使わない（敬語を保つ）話者
  "polite_marker": {"<id>": "warn"},      # 敬語の目印がない台詞を警告/失敗にする話者
  "command_endings_forbidden": ["<id>"],  # 命令形の文末を使わない話者
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# 敬語を保つ話者が使わない、くだけた常体の文末（句単位で末尾一致させる）
CASUAL_ENDINGS: tuple[str, ...] = (
    "だよ",
    "だよね",
    "だね",
    "じゃん",
    "しよう",
    "やろう",
    "だろう",
    "なの",
    "かしら",
)
# 常体の意志形（休もう・行こう等）。「しょう」の「ょ」は小書きなので一致しない
_VOLITIONAL = re.compile(r"[こごそぞとどのほぼぽもよろ]う$")
# 句全体が一致したら常体とみなす相づち
CASUAL_CLAUSES: tuple[str, ...] = ("うん", "そうだね", "だよね")
# 敬語の目印（短い台詞でもどれか一つは含むはず）
POLITE_MARKERS: tuple[str, ...] = ("です", "ます", "ません", "ましょう", "でした", "ください", "ございます")
# 命令形の文末
COMMAND_ENDINGS: tuple[str, ...] = ("しろ", "やれ", "来い", "寝ろ", "休め", "食え")

_CLAUSE_SPLIT = re.compile(r"[。！？!?\n、，,]+")
_TRAILING = "…‥・ー〜~ 　「」『』（）()♪.．"
_NARRATION = re.compile(r"[（(][^）)]*[）)]|\*[^*]+\*")


@dataclass(frozen=True)
class CheckResult:
    check: str
    ok: bool
    severity: str  # "fail" | "warn"
    speaker: str | None = None
    detail: str = ""


@dataclass
class VoiceCheckReport:
    case_id: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok and r.severity == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok and r.severity == "warn"]

    @property
    def passed(self) -> bool:
        """自動検査の必要条件を満たしたか。人物再現の合格ではない。"""
        return not self.failures


def clauses(text: str) -> list[str]:
    out = []
    for c in _CLAUSE_SPLIT.split(text):
        c = c.strip().strip(_TRAILING).strip()
        if c:
            out.append(c)
    return out


def casual_hits(text: str) -> list[str]:
    hits = []
    for c in clauses(text):
        if c in CASUAL_CLAUSES or c.endswith(CASUAL_ENDINGS) or _VOLITIONAL.search(c):
            hits.append(c)
    return hits


def command_hits(text: str) -> list[str]:
    return [c for c in clauses(text) if c.endswith(COMMAND_ENDINGS)]


def has_polite_marker(text: str) -> bool:
    return any(m in text for m in POLITE_MARKERS)


def load_cases(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _merge_phrases(base: dict[str, list[str]] | None, extra: dict[str, list[str]] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in (base or {}, extra or {}):
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def _normalize(text: str) -> str:
    return re.sub(r"[\s。、！？!?…「」]", "", text)


def check_utterances(
    utterances: list[tuple[str, str]],
    case: dict[str, Any],
    base_checks: dict[str, Any] | None = None,
) -> VoiceCheckReport:
    base = base_checks or {}
    extra = case.get("auto_checks") or {}
    cfg = {**base, **{k: v for k, v in extra.items() if k not in ("forbidden_phrases", "warn_phrases")}}
    forbidden = _merge_phrases(base.get("forbidden_phrases"), extra.get("forbidden_phrases"))
    warn = _merge_phrases(base.get("warn_phrases"), extra.get("warn_phrases"))
    report = VoiceCheckReport(case_id=case.get("id", "?"))
    add = report.results.append

    allowed = cfg.get("allowed_speakers")
    casual_forbidden = set(cfg.get("casual_endings_forbidden") or [])
    polite = cfg.get("polite_marker") or {}
    command_forbidden = set(cfg.get("command_endings_forbidden") or [])
    if not utterances:
        add(CheckResult("has_utterance", False, "fail", detail="発話がありません"))
    for speaker, text in utterances:
        if allowed and speaker not in allowed:
            add(CheckResult("allowed_speakers", False, "fail", speaker, f"不明な話者: {speaker}"))
            continue
        for phrase in forbidden.get(speaker, []) + forbidden.get("any", []):
            if phrase in text:
                add(CheckResult("forbidden_phrase", False, "fail", speaker, phrase))
        for phrase in warn.get(speaker, []) + warn.get("any", []):
            if phrase in text:
                add(CheckResult("warn_phrase", False, "warn", speaker, phrase))
        if speaker in casual_forbidden:
            for hit in casual_hits(text):
                add(CheckResult("casual_ending", False, "fail", speaker, hit))
        if polite.get(speaker) in ("warn", "fail") and not has_polite_marker(text):
            add(CheckResult("polite_marker", False, polite[speaker], speaker, text))
        if speaker in command_forbidden:
            for hit in command_hits(text):
                add(CheckResult("command_ending", False, "fail", speaker, hit))
        if cfg.get("no_narration") and _NARRATION.search(text):
            add(CheckResult("no_narration", False, "fail", speaker, "括弧書きの地の文・ト書きがあります"))
        max_chars = cfg.get("max_chars")
        if isinstance(max_chars, int) and len(text) > max_chars:
            add(CheckResult("max_chars", False, "fail", speaker, f"{len(text)}字 > {max_chars}字"))

    if cfg.get("no_duplicate_utterance"):
        norm = [_normalize(t) for _, t in utterances]
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                if norm[i] and SequenceMatcher(None, norm[i], norm[j]).ratio() >= 0.8:
                    add(CheckResult("no_duplicate_utterance", False, "fail", None, f"発話{i + 1}と発話{j + 1}がほぼ同じです"))

    speakers = {s for s, _ in utterances}
    for s in cfg.get("expected_speakers_include") or []:
        if s not in speakers:
            add(CheckResult("expected_speaker", False, "warn", s, f"呼びかけられた {s} の発話がありません"))

    if not report.results:
        add(CheckResult("all", True, "fail"))
    return report
