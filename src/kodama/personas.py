"""人物設定パック（pack.toml ＋ 人物設定ファイル）の読み込み・検証・ハッシュ・差分。

Store には依存しない。版の保存と有効化は storage 側の activate_persona_version で行う。
人物設定の版のキーは `common` とキャラクターID。
"""

from __future__ import annotations

import difflib
import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kodama.domain import MAX_CHARACTERS, Cast, CastMember, is_speaker_id

PACK_FORMAT = 1
PACK_FILE = "pack.toml"


class PackError(ValueError):
    """人物設定パックの形式・内容の誤り。"""


@dataclass(frozen=True)
class PersonaFile:
    key: str
    path: str
    body: str
    content_hash: str


@dataclass(frozen=True)
class CharacterSpec:
    id: str
    display_name: str
    persona_file: str
    address_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PersonaPack:
    root: str
    name: str
    cast: Cast
    characters: tuple[CharacterSpec, ...]
    common_file: str
    voice_cases: str | None = None
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def persona_keys(self) -> tuple[str, ...]:
        return self.cast.persona_keys

    def persona_path(self, key: str) -> Path:
        if key == "common":
            return Path(self.root) / self.common_file
        for c in self.characters:
            if c.id == key:
                return Path(self.root) / c.persona_file
        raise PackError(f"未知の人物設定キーです: {key}")

    def voice_cases_path(self) -> Path | None:
        return Path(self.root) / self.voice_cases if self.voice_cases else None


def content_hash(body: str) -> str:
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


_META_RE = re.compile(r"\A\s*<!--.*?-->[ \t]*\r?\n?", re.DOTALL)


def strip_meta(body: str) -> str:
    """先頭の `<!-- ... -->`（出典などのメタ情報）だけを取り除く。API に送る system ブロック用。
    保存する版の本文・content_hash には使わない。"""
    return _META_RE.sub("", body, count=1).lstrip("\r\n")


def _str(d: dict, key: str, where: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise PackError(f"{where}.{key} は空でない文字列が必要です")
    return v


def _relative_file(root: Path, rel: str, where: str) -> str:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise PackError(f"{where}: パック内の相対パスを指定してください: {rel}")
    if not (root / p).is_file():
        raise PackError(f"{where}: ファイルがありません: {root / p}")
    return rel


def load_pack(root: str | Path) -> PersonaPack:
    root = Path(root)
    manifest = root / PACK_FILE
    if not manifest.is_file():
        raise PackError(f"人物設定パックが見つかりません: {manifest}")
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise PackError(f"{manifest}: TOML として読めません: {e}") from None
    if data.get("pack_format") != PACK_FORMAT:
        raise PackError(f"{manifest}: 対応していない pack_format です: {data.get('pack_format')!r}")
    name = _str(data, "name", "pack")

    user = data.get("user")
    if not isinstance(user, dict):
        raise PackError("[user] が必要です")
    user_member = CastMember(id=_str(user, "id", "user"), display_name=_str(user, "display_name", "user"))

    chars_raw = data.get("characters")
    if not isinstance(chars_raw, list) or not 1 <= len(chars_raw) <= MAX_CHARACTERS:
        raise PackError(f"[[characters]] は1〜{MAX_CHARACTERS}人分必要です")
    specs: list[CharacterSpec] = []
    for i, c in enumerate(chars_raw):
        where = f"characters[{i}]"
        if not isinstance(c, dict):
            raise PackError(f"{where} の形式が不正です")
        aliases = c.get("address_aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
            raise PackError(f"{where}.address_aliases は文字列の配列にしてください")
        specs.append(
            CharacterSpec(
                id=_str(c, "id", where),
                display_name=_str(c, "display_name", where),
                persona_file=_relative_file(root, _str(c, "persona_file", where), where),
                address_aliases=tuple(aliases),
            )
        )
    for sid in [user_member.id, *(s.id for s in specs)]:
        if not is_speaker_id(sid):
            raise PackError(f"話者IDは英小文字で始まる英小文字・数字・_ の32字以内にしてください: {sid!r}")
    try:
        cast = Cast(
            user=user_member,
            characters=tuple(CastMember(id=s.id, display_name=s.display_name) for s in specs),
        )
    except ValueError as e:
        raise PackError(str(e)) from None

    common = data.get("common")
    if not isinstance(common, dict):
        raise PackError("[common] が必要です")
    common_file = _relative_file(root, _str(common, "persona_file", "common"), "common")

    voice_cases = None
    evals = data.get("evals")
    if isinstance(evals, dict) and evals.get("voice_cases"):
        voice_cases = _relative_file(root, _str(evals, "voice_cases", "evals"), "evals")

    return PersonaPack(
        root=str(root),
        name=name,
        cast=cast,
        characters=tuple(specs),
        common_file=common_file,
        voice_cases=voice_cases,
        aliases={s.id: s.address_aliases for s in specs},
    )


def load_persona_file(pack: PersonaPack, key: str) -> PersonaFile:
    path = pack.persona_path(key)
    body = path.read_text(encoding="utf-8")
    return PersonaFile(key=key, path=str(path), body=body, content_hash=content_hash(body))


def load_persona_files(pack: PersonaPack) -> dict[str, PersonaFile]:
    """全キー（common とキャラクターID）を読み込む。"""
    return {key: load_persona_file(pack, key) for key in pack.persona_keys}


def unified_diff(old_body: str, new_body: str, key: str, old_label: str = "有効版", new_label: str = "ファイル") -> str:
    lines = difflib.unified_diff(
        old_body.splitlines(keepends=True),
        new_body.splitlines(keepends=True),
        fromfile=f"{key} ({old_label})",
        tofile=f"{key} ({new_label})",
    )
    return "".join(lines)
