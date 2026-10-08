import json
import os
from types import SimpleNamespace

import pytest

from kodama.model.base import (
    ModelAPIError,
    ModelConfigError,
    ModelConnectionError,
    ModelRefusal,
    ModelRequest,
    ModelTimeout,
    redact,
)
from kodama.model.claude import ClaudeAdapter
from kodama.model.mock import MockAdapter, ScriptedAdapter, choose_speakers
from kodama.reply import REPLY_SCHEMA, parse_reply

SECRET = "sk-test-SECRET-VALUE-1234567890"


class FakeTimeout(Exception):
    pass


class FakeConn(Exception):
    pass


class FakeStatus(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


FAKE_EXC = (FakeTimeout, FakeConn, FakeStatus)


class FakeClient:
    def __init__(self, behavior):
        self.calls = []
        self.behavior = behavior
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        return self.behavior(params)


def _response(text, stop_reason="end_turn", usage=(10, 5)):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]) if usage else None,
    )


def _adapter(behavior, monkeypatch, **kw):
    monkeypatch.setenv("KODAMA_TEST_KEY", SECRET)
    made = {}

    def factory(key, timeout):
        made["key"], made["timeout"] = key, timeout
        made["client"] = FakeClient(behavior)
        return made["client"]

    a = ClaudeAdapter(api_key_env="KODAMA_TEST_KEY", client_factory=factory, exception_classes=FAKE_EXC, **kw)
    return a, made


REQ = ModelRequest(system_blocks=["共通", "蓮"], user_content="<current_input>こんにちは</current_input>", max_tokens=1200, timeout_s=30.0)
OK = json.dumps({"utterances": [{"speaker": "ren", "text": "うん。"}], "memory_candidates": []}, ensure_ascii=False)


def test_claude_params(monkeypatch):
    a, made = _adapter(lambda p: _response(OK), monkeypatch, model="claude-opus-5-5", effort="low")
    res = a.generate(REQ)
    params = made["client"].calls[0]
    assert len(made["client"].calls) == 1
    assert made["timeout"] == 30.0 and made["key"] == SECRET
    assert params["model"] == "claude-opus-5-5"
    assert params["max_tokens"] == 1200
    assert params["system"] == [{"type": "text", "text": "共通"}, {"type": "text", "text": "蓮", "cache_control": {"type": "ephemeral"}}]
    assert params["messages"] == [{"role": "user", "content": REQ.user_content}]
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": REPLY_SCHEMA}, "effort": "low"}
    assert "thinking" not in params and "fallbacks" not in params
    assert res.input_tokens == 10 and res.output_tokens == 5 and res.is_mock is False
    assert parse_reply(res.raw_text).utterances == [("ren", "うん。")]


def test_default_factory_disables_retries(monkeypatch):
    import kodama.model.claude as mod

    captured = {}

    class FakeAnthropicModule:
        class Anthropic:
            def __init__(self, **kw):
                captured.update(kw)

    monkeypatch.setitem(__import__("sys").modules, "anthropic", FakeAnthropicModule)
    mod._default_client_factory(SECRET, 12.0)
    assert captured == {"api_key": SECRET, "max_retries": 0, "timeout": 12.0}


def test_usage_unknown(monkeypatch):
    a, _ = _adapter(lambda p: _response(OK, usage=None), monkeypatch)
    res = a.generate(REQ)
    assert res.input_tokens is None and res.output_tokens is None


@pytest.mark.parametrize(
    "exc, expected, unknown",
    [
        (FakeTimeout(f"timeout {SECRET}"), ModelTimeout, True),
        (FakeConn(f"conn {SECRET}"), ModelConnectionError, True),
        (FakeStatus(f"bad key {SECRET}", 401), ModelAPIError, False),
    ],
)
def test_exception_mapping_hides_key(monkeypatch, exc, expected, unknown):
    def boom(p):
        raise exc

    a, made = _adapter(boom, monkeypatch)
    with pytest.raises(expected) as ei:
        a.generate(REQ)
    assert SECRET not in str(ei.value) and SECRET not in repr(ei.value)
    assert ei.value.__cause__ is None
    assert ei.value.outcome_unknown is unknown
    assert len(made["client"].calls) == 1  # 黙って再試行しない


def test_refusal_and_max_tokens(monkeypatch):
    a, _ = _adapter(lambda p: _response("", stop_reason="refusal"), monkeypatch)
    with pytest.raises(ModelRefusal) as ei:
        a.generate(REQ)
    assert ei.value.input_tokens == 10
    a, _ = _adapter(lambda p: _response('{"utterances": [', stop_reason="max_tokens"), monkeypatch)
    with pytest.raises(ModelAPIError):
        a.generate(REQ)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("KODAMA_NO_KEY", raising=False)
    a = ClaudeAdapter(api_key_env="KODAMA_NO_KEY", client_factory=lambda k, t: pytest.fail("呼ばれてはいけない"))
    with pytest.raises(ModelConfigError):
        a.generate(REQ)


def test_redact():
    assert redact(f"x {SECRET} y", [SECRET]) == "x [REDACTED] y"
    os.environ["KODAMA_REDACT_TEST_TOKEN"] = SECRET
    try:
        assert SECRET not in redact(f"err {SECRET}")
    finally:
        del os.environ["KODAMA_REDACT_TEST_TOKEN"]


def test_mock_is_marked_and_valid():
    m = MockAdapter()
    res = m.generate(REQ)
    assert res.is_mock and res.provider == "mock"
    parse_reply(res.raw_text)


@pytest.mark.parametrize(
    "text, speakers",
    [
        ("蓮、くたくたになったけど服を作れたよ", ["ren"]),
        ("葵、冗談でも敬語なんだね", ["aoi"]),
        ("コーヒー、おいしい。ほっとするよね", ["ren", "aoi"]),
    ],
)
def test_mock_addressing(text, speakers):
    assert choose_speakers(text) == speakers
    req = ModelRequest(system_blocks=[], user_content="", max_tokens=10, timeout_s=1, metadata={"current_input": text})
    res = MockAdapter().generate(req)
    assert [s for s, _ in parse_reply(res.raw_text).utterances] == speakers


def test_scripted_adapter():
    s = ScriptedAdapter([OK, ModelTimeout("t"), {"utterances": [{"speaker": "aoi", "text": "はい。"}]}])
    assert parse_reply(s.generate(REQ).raw_text).utterances == [("ren", "うん。")]
    with pytest.raises(ModelTimeout):
        s.generate(REQ)
    assert parse_reply(s.generate(REQ).raw_text).utterances == [("aoi", "はい。")]
    assert s.call_count == 3


def _response_cache(text, usage):
    r = _response(text)
    r.usage = SimpleNamespace(**usage)
    return r


def test_cache_control_only_on_last_system_block(monkeypatch):
    a, made = _adapter(lambda p: _response(OK), monkeypatch)
    req = ModelRequest(system_blocks=["a", "b", "c"], user_content="u", max_tokens=100, timeout_s=5.0)
    a.generate(req)
    params = made["client"].calls[0]
    assert [("cache_control" in b) for b in params["system"]] == [False, False, True]
    assert params["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in params
    assert params["messages"] == [{"role": "user", "content": "u"}]
    assert "thinking" not in params


def test_cache_usage_propagates(monkeypatch):
    usage = dict(input_tokens=10, output_tokens=5, cache_read_input_tokens=900, cache_creation_input_tokens=30)
    a, _ = _adapter(lambda p: _response_cache(OK, usage), monkeypatch)
    res = a.generate(REQ)
    assert (res.input_tokens, res.cache_read_tokens, res.cache_write_tokens) == (10, 900, 30)
    a, _ = _adapter(lambda p: _response(OK), monkeypatch)  # キャッシュ項目なし -> None
    res = a.generate(REQ)
    assert res.cache_read_tokens is None and res.cache_write_tokens is None


def test_cache_usage_on_refusal(monkeypatch):
    usage = dict(input_tokens=10, output_tokens=0, cache_read_input_tokens=7, cache_creation_input_tokens=0)
    a, _ = _adapter(lambda p: _refusal(usage), monkeypatch)
    with pytest.raises(ModelRefusal) as ei:
        a.generate(REQ)
    assert (ei.value.cache_read_tokens, ei.value.cache_write_tokens) == (7, 0)


def _refusal(usage):
    r = _response_cache("", usage)
    r.stop_reason = "refusal"
    return r
