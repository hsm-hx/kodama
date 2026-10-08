"""動作設定（TOML）。秘密（APIキー等）は設定ファイルに持たず、環境変数名だけを持つ。"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROVIDERS = ("mock", "claude")
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# キー名に含まれていたら秘密の値とみなして拒否する語（api_key_env は「環境変数の名前」なので許可）
_SECRET_WORDS = ("key", "token", "secret", "password")
_SECRET_NAME_ALLOWED = frozenset({"api_key_env"})


class ConfigError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Config:
    provider: str = "mock"
    model: str = "claude-haiku-5-5"
    effort: str = "low"
    api_key_env: str = "ANTHROPIC_API_KEY"
    db_path: str = "data/kodama.db"
    personas_dir: str = "personas"
    timezone: str = "Asia/Tokyo"
    max_tokens: int = 2000
    timeout_seconds: float = 60.0
    max_recent_messages: int = 20
    max_memories: int = 8
    max_candidates_scanned: int = 50
    max_link_hops: int = 1
    max_links_per_node: int = 10
    max_context_chars: int = 12000
    memory_auto_approve: bool = True  # モデル由来の記憶を承認済みで保存する（origin は model_candidate のまま）
    show_memory_notices: bool = True  # 自動で残した記憶を台詞の後に1行表示する

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider は {PROVIDERS} のいずれかです: {self.provider!r}")
        if self.effort not in EFFORTS:
            raise ConfigError(f"effort は {EFFORTS} のいずれかです: {self.effort!r}")
        try:
            ZoneInfo(self.timezone)
        except Exception:
            raise ConfigError(f"timezone が不正です: {self.timezone!r}") from None
        for name in ("max_tokens", "max_recent_messages", "max_memories", "max_candidates_scanned",
                     "max_links_per_node", "max_context_chars"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} は正の数にしてください")
        if self.max_link_hops not in (0, 1):
            raise ConfigError("max_link_hops は 0 か 1 です（初版は一段まで）")
        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds は正の数にしてください")

    def exportable(self) -> dict[str, Any]:
        """export に載せてよい動作設定（秘密・パスを含まない）。"""
        keys = ("timezone", "provider", "model", "max_tokens", "timeout_seconds", "effort",
                "max_recent_messages", "max_memories", "max_candidates_scanned", "max_context_chars",
                "max_link_hops", "max_links_per_node", "memory_auto_approve", "show_memory_notices")
        return {k: getattr(self, k) for k in keys}


def _check_no_secrets(values: dict[str, Any]) -> None:
    for k in values:
        if k in _SECRET_NAME_ALLOWED:
            continue
        if any(w in k.lower() for w in _SECRET_WORDS):
            raise ConfigError(
                f"設定ファイルに秘密らしき項目 {k!r} があります。APIキーは環境変数で設定してください。"
            )


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    values: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"設定ファイルが見つかりません: {p}")
        with open(p, "rb") as fh:
            try:
                values = tomllib.load(fh)
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(f"設定ファイルを読めません: {e}") from None
    _check_no_secrets(values)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(f"未知の設定項目です: {unknown}")
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    cfg = Config(**values)
    cfg.validate()
    return cfg
