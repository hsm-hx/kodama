from pathlib import Path

from kodama.personas import PERSONA_KEYS, content_hash, load_persona_files, unified_diff

ROOT = Path(__file__).resolve().parents[1]


def test_load_repo_personas():
    files = load_persona_files(ROOT / "personas")
    assert set(files) == set(PERSONA_KEYS)
    for f in files.values():
        assert f.body.strip()
        assert f.content_hash == content_hash(f.body)
    assert "あなた" in files["common"].body
    assert "俺" in files["ren"].body  # 「使わない」規則として記載
    assert "わたし" in files["aoi"].body


def test_hash_and_diff(tmp_path):
    for k in PERSONA_KEYS:
        (tmp_path / f"{k}.md").write_text(f"# {k}\n本文\n", encoding="utf-8")
    a = load_persona_files(tmp_path)
    (tmp_path / "ren.md").write_text("# ren\n本文を変更\n", encoding="utf-8")
    b = load_persona_files(tmp_path)
    assert a["ren"].content_hash != b["ren"].content_hash
    assert a["aoi"].content_hash == b["aoi"].content_hash
    d = unified_diff(a["ren"].body, b["ren"].body, "ren")
    assert "-本文" in d and "+本文を変更" in d
    assert unified_diff("x\n", "x\n", "ren") == ""


def test_strip_meta():
    from kodama.personas import strip_meta

    body = "<!--\nkey: x\nnote: n\n-->\n\n# 本文\n<!-- 本文中のコメントは残す -->\n"
    assert strip_meta(body) == "# 本文\n<!-- 本文中のコメントは残す -->\n"
    assert strip_meta("# 先頭にメタなし\n") == "# 先頭にメタなし\n"
