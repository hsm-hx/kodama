"""会話モデル接続の境界。UI・記憶層はこのモジュールの型だけを扱う。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class ModelRequest:
    system_blocks: list[str]
    user_content: str
    max_tokens: int
    timeout_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResult:
    raw_text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    is_mock: bool = False


class ModelAdapter(Protocol):
    provider: str
    model: str
    is_mock: bool

    def generate(self, request: ModelRequest) -> ModelResult: ...


class ModelError(Exception):
    """モデル呼び出しの失敗。メッセージに秘密を含めないこと。"""

    # 送信後に結果が分からない（課金の有無も不明）場合 True
    outcome_unknown: bool = False
    # 応答が返ったが使えなかった場合に、取得できた利用量（不明なら None）
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelTimeout(ModelError):
    outcome_unknown = True


class ModelConnectionError(ModelError):
    # 接続失敗は送信前か後か区別できないため、不明として扱う
    outcome_unknown = True


class ModelAPIError(ModelError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ModelRefusal(ModelError):
    pass


class ModelConfigError(ModelError):
    """キー未設定・SDK未導入など、送信前に止めた失敗。"""


_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET")


def redact(text: str, secrets: Iterable[str] | None = None) -> str:
    """文字列に秘密の値が含まれていれば伏せる。

    secrets を省略すると、名前に KEY/TOKEN/SECRET を含む環境変数の値（8文字以上）を対象にする。
    """
    if secrets is None:
        secrets = [
            v for k, v in os.environ.items() if any(h in k.upper() for h in _SECRET_ENV_HINTS) and v and len(v) >= 8
        ]
    out = text
    for s in sorted(set(secrets), key=len, reverse=True):
        if s and len(s) >= 8:
            out = out.replace(s, "[REDACTED]")
    return out
