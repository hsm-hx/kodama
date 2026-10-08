"""人物設定ファイル（personas/*.md）の読み込み・ハッシュ・差分。

Store には依存しない。版の保存と有効化は storage 側の activate_persona_version で行う。
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path

PERSONA_KEYS: tuple[str, ...] = ("common", "ren", "aoi")


@dataclass(frozen=True)
class PersonaFile:
    key: str
    path: str
    body: str
    content_hash: str


def content_hash(body: str) -> str:
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_persona_file(directory: str | Path, key: str) -> PersonaFile:
    if key not in PERSONA_KEYS:
        raise ValueError(f"未知の人物設定キーです: {key}")
    path = Path(directory) / f"{key}.md"
    body = path.read_text(encoding="utf-8")
    return PersonaFile(key=key, path=str(path), body=body, content_hash=content_hash(body))


def load_persona_files(directory: str | Path) -> dict[str, PersonaFile]:
    """全キーを読み込む。欠けていれば FileNotFoundError。"""
    return {key: load_persona_file(directory, key) for key in PERSONA_KEYS}


def unified_diff(old_body: str, new_body: str, key: str, old_label: str = "有効版", new_label: str = "ファイル") -> str:
    lines = difflib.unified_diff(
        old_body.splitlines(keepends=True),
        new_body.splitlines(keepends=True),
        fromfile=f"{key} ({old_label})",
        tofile=f"{key} ({new_label})",
    )
    return "".join(lines)
