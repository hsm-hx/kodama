"""APIキーが画面・保存・例外・export に出ないこと（ネットワーク不要、偽クライアント使用）。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kodama import cli
from kodama.config import ConfigError, load_config
from kodama.model.claude import ClaudeAdapter

SECRET = "sk-ant-api03-DUMMY-NOT-A-REAL-KEY-0123456789"
REPO_PERSONAS = Path(__file__).resolve().parent.parent / "personas"


class FakeTimeout(Exception):
    pass


class FakeConn(Exception):
    pass


class FakeStatus(Exception):
    def __init__(self, msg, status_code=401):
        super().__init__(msg)
        self.status_code = status_code


class LeakyClient:
    """例外メッセージにキーを含めてくる最悪の偽クライアント。"""

    def __init__(self, key, exc):
        self.key = key
        self.exc = exc
        self.messages = self

    def create(self, **params):
        raise self.exc(f"bad key {self.key}")


def _adapter(exc):
    return ClaudeAdapter(
        model="claude-opus-5-5",
        client_factory=lambda key, timeout: LeakyClient(key, exc),
        exception_classes=(FakeTimeout, FakeConn, FakeStatus),
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.chdir(tmp_path)
    personas = tmp_path / "personas"
    shutil.copytree(REPO_PERSONAS, personas)
    cfg = tmp_path / "kodama.toml"
    cfg.write_text(f'provider = "claude"\npersonas_dir = "{personas}"\n', encoding="utf-8")
    return tmp_path, cfg


@pytest.mark.parametrize("exc", [FakeStatus, FakeTimeout, FakeConn, RuntimeError])
def test_key_never_shown_stored_or_exported(env, exc):
    tmp, cfg = env
    out: list[str] = []
    lines = iter(["y", "こんばんは", "/history", f"/export {tmp / 'e.json'}", "/quit"])
    code = cli.main(["--config", str(cfg), "--db", str(tmp / "k.db")],
                    lambda p: next(lines), out.append, lambda c: _adapter(exc))
    assert code == 0
    text = "\n".join(out)
    assert "失敗" in text or "エラー" in text or "タイムアウト" in text or "接続" in text
    assert SECRET not in text
    assert not any(l.startswith(("蓮「", "葵「")) for l in out)
    for f in tmp.iterdir():
        if f.is_file():
            assert SECRET.encode() not in f.read_bytes(), f.name


def test_config_file_with_key_is_rejected_without_echo(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(f'api_key = "{SECRET}"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert SECRET not in str(e.value)
    out: list[str] = []
    code = cli.main(["--config", str(p), "--db", str(tmp_path / "k.db")], lambda p: "", out.append)
    assert code == 2 and SECRET not in "\n".join(out)


def test_missing_key_is_config_error_and_mock_still_works(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    personas = tmp_path / "personas"
    shutil.copytree(REPO_PERSONAS, personas)
    cfg = tmp_path / "kodama.toml"
    cfg.write_text(f'provider = "claude"\npersonas_dir = "{personas}"\n', encoding="utf-8")
    out: list[str] = []
    lines = iter(["y", "こんばんは", "/quit"])
    code = cli.main(["--config", str(cfg), "--db", str(tmp_path / "k.db")], lambda p: next(lines), out.append,
                    lambda c: ClaudeAdapter(model=c.model, client_factory=lambda k, t: None))
    text = "\n".join(out)
    assert code == 0 and "ANTHROPIC_API_KEY" in text and "設定の問題" in text
