import json

import pytest

from kodama.reply import InvalidReply, reply_schema
from kodama.reply import parse_reply as _parse_reply
from seed import CAST, PACK

IDS = CAST.character_ids
REPLY_SCHEMA = reply_schema(IDS)
CHARACTERS = [{"id": c.id, "display_name": c.display_name, "aliases": list(c.address_aliases)} for c in PACK.characters]


def parse_reply(raw):
    return _parse_reply(raw, IDS)



def _raw(utts, cands=None):
    d = {"utterances": [{"speaker": s, "text": t} for s, t in utts]}
    if cands is not None:
        d["memory_candidates"] = cands
    return json.dumps(d, ensure_ascii=False)


def test_two_speakers():
    r = parse_reply(_raw([("ren", "頑張ったね。"), ("aoi", "今日は十分です。")], []))
    assert r.utterances == [("ren", "頑張ったね。"), ("aoi", "今日は十分です。")]
    assert r.candidates == [] and r.warnings == []


def test_single_speaker_and_fence():
    raw = "```json\n" + _raw([("aoi", "はい。")]) + "\n```"
    assert parse_reply(raw).utterances == [("aoi", "はい。")]


@pytest.mark.parametrize(
    "raw",
    [
        "蓮「こんにちは」",  # JSONではない
        "前置きです\n" + _raw([("ren", "うん")]),  # 余計な地の文
        _raw([("user", "なりすまし")]),  # 不正な話者
        _raw([("narrator", "地の文")]),
        _raw([("ren", "   ")]),  # 空の発話
        _raw([]),  # 発話なし
        _raw([("ren", "a")] * 5),  # 多すぎる
        _raw([("ren", "あ" * 1001)]),
        json.dumps({"utterances": [{"speaker": "ren", "text": "x"}], "command": "send"}),
        json.dumps([{"speaker": "ren", "text": "x"}]),
        '{"utterances": [{"speaker": "ren", "text": "途中で',
    ],
)
def test_invalid(raw):
    with pytest.raises(InvalidReply):
        parse_reply(raw)


def test_bad_candidates_dropped_with_warning():
    cands = [
        {"body": "あなたは服を作った", "kind": "user_stated", "perspective": None, "subjects": ["あなた"], "tags": ["服"]},
        {"body": "嬉しそうだった", "kind": "character_view", "perspective": None, "subjects": [], "tags": []},
        {"body": "x", "kind": "fact", "perspective": None, "subjects": [], "tags": []},
        {"body": "", "kind": "imagination", "perspective": None, "subjects": [], "tags": []},
        {"body": "葵の受け取り", "kind": "character_view", "perspective": "aoi", "subjects": [], "tags": []},
    ]
    r = parse_reply(_raw([("ren", "うん。")], cands))
    assert [c["body"] for c in r.candidates] == ["あなたは服を作った", "葵の受け取り"]
    assert len(r.warnings) == 3


def test_schema_constant_shape():
    assert REPLY_SCHEMA["additionalProperties"] is False
    assert REPLY_SCHEMA["properties"]["utterances"]["items"]["properties"]["speaker"]["enum"] == ["ren", "aoi"]
