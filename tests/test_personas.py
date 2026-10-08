from pathlib import Path

import pytest

from kodama.personas import PackError, content_hash, load_pack, load_persona_files, strip_meta, unified_diff
from seed import EXAMPLE_PACK_DIR


def _write_pack(root: Path, characters=(("ren", "蓮"), ("aoi", "葵")), user=("user", "あなた"), extra="") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "common.md").write_text("# common\n本文\n", encoding="utf-8")
    chars = ""
    for cid, name in characters:
        (root / f"{cid}.md").write_text(f"# {cid}\n本文\n", encoding="utf-8")
        chars += f'\n[[characters]]\nid = "{cid}"\ndisplay_name = "{name}"\npersona_file = "{cid}.md"\n'
    (root / "pack.toml").write_text(
        f'pack_format = 1\nname = "t"\n[user]\nid = "{user[0]}"\ndisplay_name = "{user[1]}"\n{chars}'
        f'\n[common]\npersona_file = "common.md"\n{extra}',
        encoding="utf-8",
    )
    return root


def test_load_example_pack():
    pack = load_pack(EXAMPLE_PACK_DIR)
    assert pack.cast.user_id == "user"
    assert pack.cast.character_ids == ("ren", "aoi")
    assert pack.persona_keys == ("common", "ren", "aoi")
    files = load_persona_files(pack)
    assert set(files) == set(pack.persona_keys)
    for f in files.values():
        assert f.body.strip()
        assert f.content_hash == content_hash(f.body)
    assert pack.voice_cases_path() is not None and pack.voice_cases_path().is_file()


def test_pack_validation(tmp_path):
    assert load_pack(_write_pack(tmp_path / "ok")).cast.character_ids == ("ren", "aoi")
    with pytest.raises(PackError):
        load_pack(tmp_path / "missing")
    with pytest.raises(PackError):
        load_pack(_write_pack(tmp_path / "badid", characters=(("Ren", "蓮"),)))
    with pytest.raises(PackError):
        load_pack(_write_pack(tmp_path / "dup", characters=(("ren", "蓮"), ("ren", "葵"))))
    with pytest.raises(PackError):
        load_pack(_write_pack(tmp_path / "userdup", characters=(("user", "蓮"),)))
    with pytest.raises(PackError):
        load_pack(_write_pack(tmp_path / "toomany", characters=tuple((f"c{i}", f"n{i}") for i in range(5))))
    root = _write_pack(tmp_path / "nofile")
    (root / "aoi.md").unlink()
    with pytest.raises(PackError):
        load_pack(root)
    root = _write_pack(tmp_path / "escape")
    text = (root / "pack.toml").read_text(encoding="utf-8").replace('persona_file = "aoi.md"', 'persona_file = "../aoi.md"')
    (root / "pack.toml").write_text(text, encoding="utf-8")
    with pytest.raises(PackError):
        load_pack(root)


def test_hash_and_diff(tmp_path):
    pack = load_pack(_write_pack(tmp_path / "p"))
    a = load_persona_files(pack)
    (tmp_path / "p" / "ren.md").write_text("# ren\n本文を変更\n", encoding="utf-8")
    b = load_persona_files(pack)
    assert a["ren"].content_hash != b["ren"].content_hash
    assert a["aoi"].content_hash == b["aoi"].content_hash
    d = unified_diff(a["ren"].body, b["ren"].body, "ren")
    assert "-本文" in d and "+本文を変更" in d
    assert unified_diff("x\n", "x\n", "ren") == ""


def test_strip_meta():
    body = "<!--\nkey: x\nnote: n\n-->\n\n# 本文\n<!-- 本文中のコメントは残す -->\n"
    assert strip_meta(body) == "# 本文\n<!-- 本文中のコメントは残す -->\n"
    assert strip_meta("# 先頭にメタなし\n") == "# 先頭にメタなし\n"
