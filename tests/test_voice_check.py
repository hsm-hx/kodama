"""自動検査関数そのものの試験。手書きサンプルで判定を確かめる。

ここでの合格は検査関数が期待どおり動くことの確認であり、
モデル（モック・実モデルとも）が人物を再現できたことの証拠ではない。
"""

import pytest

from kodama.voice_check import casual_hits, check_utterances, load_cases
from seed import PACK

# 検査関数の試験用の規則（話者IDで書く）。ren は命令形を使わない、aoi は常に丁寧語。
BASE = {
    "allowed_speakers": ["ren", "aoi"],
    "forbidden_phrases": {"ren": ["俺", "よく作ったな"], "any": []},
    "warn_phrases": {"ren": ["あおちゃん"]},
    "casual_endings_forbidden": ["aoi"],
    "polite_marker": {"aoi": "warn"},
    "command_endings_forbidden": ["ren"],
    "no_duplicate_utterance": True,
    "no_narration": True,
}
CASE = {"id": "t", "auto_checks": {}}


def _check(utts, case=CASE):
    return check_utterances(utts, case, BASE)


def test_example_pack_cases_file_shape():
    data = load_cases(PACK.voice_cases_path())
    assert data["cases"]
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids))
    for c in data["cases"]:
        assert c["input"] and c["intent"] and c["human_review"]
    assert set(data["base_checks"]["allowed_speakers"]) == set(PACK.cast.character_ids)


@pytest.mark.parametrize(
    "text",
    [
        "うん、そうだね。今日は休もう",
        "一緒にやろう",
        "大丈夫だよ",
        "それ、ずるいじゃん",
        "そうなの？",
        "どうかしら",
        "わたしもそう思うだね",
    ],
)
def test_casual_detected(text):
    assert casual_hits(text)


@pytest.mark.parametrize(
    "text",
    [
        "はい、その判断で問題ありません。今日は休みましょう",
        "それは大切なことなのです。",
        "一緒に確認しましょう。",
        "撤回しても遅いですよ。記録します",
        "少しだけ、ミルクを入れても良さそうです",
        "不安があることと、危険であることは同一ではありません。順番に確認しましょう",
        "蓮くんも、そう考えているようですね。",
        "でしょうか。",
    ],
)
def test_polite_not_flagged(text):
    assert casual_hits(text) == []


def test_good_sample_passes():
    r = _check([("ren", "……うん。あったかいね"), ("aoi", "香りが強めです。わたしは、少しだけミルクを入れても良さそうだと考えます")])
    assert r.passed, r.failures


def test_casual_ending_fails_only_for_configured_speaker():
    r = _check([("ren", "ほっとするね"), ("aoi", "うん、ほっとするね")])
    assert {(f.check, f.speaker) for f in r.failures} >= {("casual_ending", "aoi")}
    assert not any(f.check == "casual_ending" and f.speaker == "ren" for f in r.failures)


def test_polite_marker_warns():
    r = _check([("aoi", "なるほど")])
    assert any(w.check == "polite_marker" for w in r.warnings)
    strict = check_utterances([("aoi", "なるほど")], {"id": "s", "auto_checks": {"polite_marker": {"aoi": "fail"}}}, BASE)
    assert any(f.check == "polite_marker" for f in strict.failures)


def test_duplicate_detected():
    r = _check([("ren", "今日はゆっくり休みましょう。"), ("aoi", "今日はゆっくり休みましょうね。")])
    assert any(f.check == "no_duplicate_utterance" for f in r.failures)


def test_forbidden_phrases():
    r = _check([("ren", "俺、見てたよ。よく作ったな")])
    assert {f.detail for f in r.failures} >= {"俺", "よく作ったな"}
    assert _check([("ren", "……頑張ったね。よく作ったね")]).passed


def test_case_level_forbidden():
    case = {"id": "c", "auto_checks": {"forbidden_phrases": {"any": ["承知しました"]}}}
    assert not _check([("aoi", "承知しました。では続きです。")], case).passed
    assert _check([("aoi", "それで、さっきの散歩の話ですが、あの坂は少し急でしたね。")], case).passed


def test_unknown_speaker_and_narration():
    r = _check([("narrator", "二人は顔を見合わせた"), ("ren", "（うなずいて）……うん")])
    checks = {f.check for f in r.failures}
    assert {"allowed_speakers", "no_narration"} <= checks


def test_command_ending():
    r = _check([("ren", "早く寝ろ")])
    assert any(f.check == "command_ending" for f in r.failures)
    assert not any(f.check == "command_ending" for f in _check([("aoi", "早く寝ろ")]).failures)
