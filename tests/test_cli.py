from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kodama import cli
from kodama.model.base import ModelTimeout
from kodama.model.mock import MockAdapter, ScriptedAdapter

REPO_PERSONAS = Path(__file__).resolve().parent.parent / "personas"


class Session:
    """main() を入力列で駆動し、出力行を集める。"""

    def __init__(self, tmp_path, provider="mock"):
        self.tmp = tmp_path
        self.personas = tmp_path / "personas"
        if not self.personas.exists():
            shutil.copytree(REPO_PERSONAS, self.personas)
        self.db = tmp_path / "data" / "k.db"
        self.config = tmp_path / "kodama.toml"
        self.config.write_text(f'provider = "{provider}"\npersonas_dir = "{self.personas}"\n', encoding="utf-8")

    def run(self, lines, adapter=None, extra=()):
        out: list[str] = []
        feed = iter(lines)

        def input_fn(prompt):
            try:
                return next(feed)
            except StopIteration:
                raise EOFError

        made = []

        def factory(config):
            made.append(config)
            return adapter if adapter is not None else MockAdapter()

        code = cli.main(["--config", str(self.config), "--db", str(self.db), *extra], input_fn, out.append, factory)
        return code, out, made


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_mock_conversation_history_and_resume(tmp_path):
    s = Session(tmp_path)
    code, out, _ = s.run(["蓮、ただいま", "コーヒー、おいしい。ほっとするよね", "/history", "/quit"])
    assert code == 0
    text = "\n".join(out)
    assert "モックモード" in text and "[モック応答]" in text
    assert "蓮「" in text and "葵「" in text
    assert any(line.startswith("操作:") for line in out)

    code, out2, _ = s.run(["/history", "/quit"])
    hist = [l for l in out2 if l.startswith("[")]
    # 起動時の直前表示＋/history の両方に、元の順序のまま出る
    full = hist[-5:]
    assert "あなた「蓮、ただいま」" in full[0]
    assert "蓮「" in full[1]
    assert "あなた「コーヒー、おいしい。ほっとするよね」" in full[2]
    assert "蓮「" in full[3] and "葵「" in full[4]


def test_session_new_list_resume(tmp_path):
    s = Session(tmp_path)
    s.run(["一つ目の話", "/quit"])
    code, out, _ = s.run(["/session new 二つ目", "二つ目の話", "/session list", "/quit"])
    listing = [l for l in out if "件)" in l]
    assert len(listing) == 2
    first_id = listing[0].split()[1]
    code, out, _ = s.run([f"/session resume {first_id}", "/history", "/quit"])
    text = "\n".join(out)
    assert "一つ目の話" in text.split("セッションを再開しました")[1]


def test_errors_are_operations_not_dialogue(tmp_path):
    s = Session(tmp_path)
    adapter = ScriptedAdapter([ModelTimeout("応答がタイムアウトしました。")])
    code, out, _ = s.run(["こんばんは", "/history", "/quit"], adapter=adapter)
    timeout_lines = [l for l in out if "タイムアウト" in l]
    assert timeout_lines and timeout_lines[0].startswith("操作:")
    assert all(l.startswith(("操作:", "      ")) for l in timeout_lines)
    assert not any(l.startswith(("蓮「", "葵「")) for l in out)
    assert any("未応答" in l for l in out)
    assert any("自動では再送しません" in l for l in out)
    assert adapter.call_count == 1


def test_works_without_key_or_model(tmp_path):
    s = Session(tmp_path)
    s.config.write_text(f'personas_dir = "{s.personas}"\n', encoding="utf-8")
    code, out, _ = s.run(["ねえ、聞いて", "/memory add あなたはねこが好きだと話した --tags ねこ", "/memory list",
                          "/history", "/context ねこの話", "/quit"])
    text = "\n".join(out)
    assert code == 0 and "登録しました" in text and "あなたはねこが好き" in text
    assert "送信はしていません" in text


def test_memory_commands_approve_reject_revise(tmp_path):
    s = Session(tmp_path)
    code, out, _ = s.run([
        "朝のコーヒーがおいしい",
        "/memory add あなたは朝にコーヒーを飲むと話した --tags コーヒー --about コーヒー",
        "/memory list",
        "/quit",
    ])
    line = next(l for l in out if l.startswith("操作: 登録しました: "))
    mem_id = line.split("登録しました: ")[1].split()[0]
    code, out, _ = s.run([f"/memory revise {mem_id} あなたは夜にコーヒーを飲むと話した", "/memory list all",
                          f"/memory show {mem_id}", "/context コーヒー", "/quit"])
    text = "\n".join(out)
    assert "訂正済み(旧版)" in text and "夜にコーヒー" in text
    ctx = text.split("次の送信内容")[1]
    assert "夜にコーヒー" in ctx and "朝にコーヒー" not in ctx.split("除外")[0]


def test_real_api_requires_consent(tmp_path):
    s = Session(tmp_path, provider="claude")
    code, out, made = s.run(["n", "n"])
    assert code == 0 and made == []  # 同意しなければアダプタも作らない
    assert any("送る内容" in l for l in out)
    code, out, made = s.run(["n", "y", "やあ", "/quit"])
    assert made and made[0].provider == "mock"
    assert "[モック応答]" in out


def test_persona_change_needs_approval(tmp_path):
    s = Session(tmp_path)
    s.run(["/quit"])
    ren = s.personas / "ren.md"
    ren.write_text(ren.read_text(encoding="utf-8") + "\n- 追記: 散歩が好き。\n", encoding="utf-8")
    code, out, _ = s.run(["/persona status", "/persona diff ren", "/persona approve ren", "/persona history ren", "/quit"])
    text = "\n".join(out)
    assert "未承認の変更があります: ren" in text
    assert "+- 追記: 散歩が好き。" in text
    assert "新しい版" in text
    assert "[retired]" in text and "[active]" in text


def test_export_import_commands(tmp_path):
    s = Session(tmp_path)
    exp = tmp_path / "e.json"
    code, out, _ = s.run(["こんにちは", f"/export {exp}", f"/import verify {exp}", f"/import {exp} {s.db}",
                          f"/import {exp} {tmp_path / 'new.db'}", "/quit"])
    text = "\n".join(out)
    assert "書き出しました" in text and "検証OK" in text
    assert "currently in use" in text
    assert (tmp_path / "new.db").exists() and "--db" in text


def test_unknown_command_and_model_text_never_run_commands(tmp_path):
    s = Session(tmp_path)
    adapter = ScriptedAdapter([{"utterances": [{"speaker": "ren", "text": "/export leak.json"}],
                                "memory_candidates": []}])
    code, out, _ = s.run(["/nope", "なにか", "/quit"], adapter=adapter)
    assert any("不明なコマンド" in l for l in out)
    assert '蓮「/export leak.json」' in out
    assert not (tmp_path / "leak.json").exists()


def test_model_override_shown_at_real_api_confirmation(tmp_path):
    s = Session(tmp_path, provider="claude")
    code, out, made = s.run(["n", "n"], extra=("--model", "claude-haiku-5-5"))
    assert code == 0 and made == []
    assert any("モデル claude-haiku-5-5" in l for l in out)
    code, out, made = s.run(["n", "n"])
    assert any("モデル claude-sonnet-5-5" in l for l in out)  # 既定


def test_model_override_reaches_adapter_config(tmp_path):
    s = Session(tmp_path, provider="claude")
    code, out, made = s.run(["y", "/quit"], extra=("--model", "claude-opus-5-5"))
    assert made and made[0].model == "claude-opus-5-5"
    assert any("実APIモード" in l for l in out) or code == 0


def test_usage_line_with_cache(tmp_path):
    from kodama.model.base import ModelResult
    import json

    reply = json.dumps({"utterances": [{"speaker": "ren", "text": "うん。"}], "memory_candidates": []}, ensure_ascii=False)
    adapter = ScriptedAdapter([ModelResult(reply, "anthropic", "m", 100, 12, 800, 40)])
    s = Session(tmp_path)
    code, out, _ = s.run(["やあ", "/quit"], adapter=adapter)
    line = next(l for l in out if "利用量" in l)
    assert "入力 940（うちキャッシュ読込 800 / 書込 40）・出力 12 トークン" in line
