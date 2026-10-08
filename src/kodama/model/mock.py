"""APIキー不要のモック。返答は固定の決定的な文で、実際のLLM応答ではない。

表示側は ModelResult.is_mock を見て「モック」と明示すること。
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Iterable

from .base import ModelRequest, ModelResult

_ADDRESS_REN = re.compile(r"^\s*(蓮|れん|レン)\s*[、,，!！?？]")
_ADDRESS_AOI = re.compile(r"^\s*(葵|あおい|アオイ)\s*[、,，!！?？]")

_CURRENT_INPUT_RE = re.compile(r"<current_input>\s*(.*?)\s*</current_input>", re.DOTALL)


def _current_input(request: ModelRequest) -> str:
    text = request.metadata.get("current_input")
    if isinstance(text, str):
        return text
    m = _CURRENT_INPUT_RE.search(request.user_content)
    return m.group(1) if m else request.user_content


def choose_speakers(user_text: str) -> list[str]:
    """モックの話者選択（決定的）。一人への明確な呼びかけならその人だけ。"""
    if _ADDRESS_REN.match(user_text):
        return ["ren"]
    if _ADDRESS_AOI.match(user_text):
        return ["aoi"]
    # 短い相づち程度なら一人だけ（文字数の偶奇で決める）
    stripped = user_text.strip()
    if len(stripped) <= 6:
        return ["ren"] if len(stripped) % 2 == 0 else ["aoi"]
    return ["ren", "aoi"]


_MOCK_TEXT = {
    "ren": "……モックの蓮の台詞です。",
    "aoi": "モックの葵の台詞です。",
}


class MockAdapter:
    provider = "mock"
    is_mock = True

    def __init__(self, model: str = "mock"):
        self.model = model
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        speakers = choose_speakers(_current_input(request))
        payload = {
            "utterances": [{"speaker": s, "text": _MOCK_TEXT[s]} for s in speakers],
            "memory_candidates": [],
        }
        return ModelResult(
            raw_text=json.dumps(payload, ensure_ascii=False),
            provider=self.provider,
            model=self.model,
            input_tokens=None,
            output_tokens=None,
            stop_reason="end_turn",
            is_mock=True,
        )


class ScriptedAdapter:
    """テスト用。キューの順に raw_text（str / dict）か例外を返し、受け取ったリクエストを記録する。"""

    provider = "scripted"
    is_mock = True

    def __init__(self, script: Iterable[Any] = (), model: str = "scripted"):
        self.model = model
        self.queue: deque[Any] = deque(script)
        self.requests: list[ModelRequest] = []

    def push(self, item: Any) -> None:
        self.queue.append(item)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def generate(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("ScriptedAdapter: 応答キューが空です")
        item = self.queue.popleft()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ModelResult):
            return item
        raw = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return ModelResult(raw_text=raw, provider=self.provider, model=self.model, stop_reason="end_turn", is_mock=True)
