"""Claude API（Messages API）への実接続 adapter。

- 1回の generate につき messages.create を1回だけ呼ぶ。SDKの自動リトライは max_retries=0 で無効化。
- server-side fallbacks は送らない（別モデルでの再実行＝黙った再試行・追加課金になり得るため）。
- APIキーは環境変数から読むだけで、保存・ログ・例外文字列には出さない。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..reply import REPLY_SCHEMA
from .base import (
    ModelAPIError,
    ModelConfigError,
    ModelConnectionError,
    ModelError,
    ModelRefusal,
    ModelRequest,
    ModelResult,
    ModelTimeout,
    redact,
)

DEFAULT_MODEL = "claude-sonnet-5-5"


def _default_client_factory(api_key: str, timeout_s: float) -> Any:
    try:
        import anthropic  # 遅延import: モック利用時は不要
    except ImportError:
        raise ModelConfigError(
            "anthropic パッケージが入っていません。`uv sync --extra claude` で導入してください。"
        ) from None
    return anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout_s)


def _int_or_none(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _exception_classes() -> tuple[type, type, type] | None:
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.APIStatusError


class ClaudeAdapter:
    provider = "anthropic"
    is_mock = False

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key_env: str = "ANTHROPIC_API_KEY",
        effort: str | None = "low",
        client_factory: Callable[[str, float], Any] | None = None,
        exception_classes: tuple[type, type, type] | None = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.effort = effort
        self._client_factory = client_factory or _default_client_factory
        self._exception_classes = exception_classes

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key.strip():
            raise ModelConfigError(f"環境変数 {self.api_key_env} が設定されていません。")
        return key

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": REPLY_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        system: list[dict[str, Any]] = [{"type": "text", "text": b} for b in request.system_blocks]
        if system:
            # 固定部分（人物設定＋出力規則）の末尾だけにブレークポイントを置く。
            # トップレベルの自動 cache_control は、毎回変わる user content 側に付くので使わない。
            system[-1]["cache_control"] = {"type": "ephemeral"}
        return {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": request.user_content}],
            "output_config": output_config,
        }

    def generate(self, request: ModelRequest) -> ModelResult:
        key = self._api_key()
        try:
            return self._generate(request, key)
        except ModelError as e:
            # 念のためメッセージからキーを伏せ直す
            e.args = tuple(redact(str(a), [key]) if isinstance(a, str) else a for a in e.args)
            raise

    def _generate(self, request: ModelRequest, key: str) -> ModelResult:
        client = self._client_factory(key, request.timeout_s)
        classes = self._exception_classes or _exception_classes()
        params = self.build_params(request)
        if classes is None:
            response = client.messages.create(**params)
        else:
            timeout_cls, conn_cls, status_cls = classes
            try:
                response = client.messages.create(**params)
            except timeout_cls:
                raise ModelTimeout("応答がタイムアウトしました。送信済みかどうか・課金の有無は不明です。") from None
            except conn_cls:
                raise ModelConnectionError("接続に失敗しました。送信済みかどうか・課金の有無は不明です。") from None
            except status_cls as e:
                status = getattr(e, "status_code", None)
                raise ModelAPIError(f"APIがエラーを返しました（HTTP {status}）。", status=status) from None

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
        cache_read = _int_or_none(getattr(usage, "cache_read_input_tokens", None)) if usage is not None else None
        cache_write = _int_or_none(getattr(usage, "cache_creation_input_tokens", None)) if usage is not None else None
        stop_reason = getattr(response, "stop_reason", None)
        err: ModelError | None = None
        if stop_reason == "refusal":
            err = ModelRefusal("モデルが応答を辞退しました。")
        elif stop_reason == "max_tokens":
            err = ModelAPIError("出力が上限（max_tokens）で切れました。返答は保存しません。")
        if err is not None:
            # 応答自体は返っているので利用量は記録できる
            err.input_tokens = input_tokens if isinstance(input_tokens, int) else None
            err.output_tokens = output_tokens if isinstance(output_tokens, int) else None
            err.cache_read_tokens, err.cache_write_tokens = cache_read, cache_write
            raise err
        text = "".join(
            getattr(b, "text", "") for b in (getattr(response, "content", None) or []) if getattr(b, "type", None) == "text"
        )
        return ModelResult(
            raw_text=text,
            provider=self.provider,
            model=getattr(response, "model", None) or self.model,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            stop_reason=stop_reason,
            is_mock=False,
        )
