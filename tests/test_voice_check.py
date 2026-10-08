"""自動検査関数そのものの試験。手書きサンプルで判定を確かめる。

ここでの合格は検査関数が期待どおり動くことの確認であり、
モデル（モック・実モデルとも）が人物を再現できたことの証拠ではない。
"""

import pytest

from kodama.voice_check import check_utterances, load_cases, aoi_casual_hits

DATA = load_cases()
BASE = DATA["base_checks"]
CASES = {c["id"]: c for c in DATA["cases"]}


def _check(case_id, utts):
    return check_utterances(utts, CASES[case_id], BASE)


def test_cases_file_shape():
    assert len(DATA["cases"]) >= 8
    ids = [c["id"] for c in DATA["cases"]]
    assert len(ids) == len(set(ids))
    for c in DATA["cases"]:
        assert c["input"] and c["intent"] and c["human_review"] and c["source"]


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
def test_aoi_casual_detected(text):
    assert aoi_casual_hits(text)


@pytest.mark.parametrize(
    "text",
    [
        "はい、その判断で問題ありません。今日は休みましょう",
        "それは大切なことなのです。",
        "一緒に確認しましょう。",
        "撤回しても遅いですよ。記録します",
        "少しだけ、ミルクを入れても良さそうです",
        "不安があることと、危険であることは同一ではありません。順番に確認しましょう",
        "れんも、そう考えているようですね。",
        "でしょうか。",
    ],
)
def test_aoi_polite_not_flagged(text):
    assert aoi_casual_hits(text) == []


def test_good_sample_passes_case01():
    r = _check("case01_coffee", [("ren", "……うん。あったかいね"), ("aoi", "香りが強めです。わたしは、少しだけミルクを入れても良さそうだと考えます")])
    assert r.passed, r.failures


def test_bad_sample_case01_casual_and_duplicate():
    r = _check("case01_coffee", [("ren", "ほっとするね"), ("aoi", "うん、ほっとするね")])
    checks = {f.check for f in r.failures}
    assert "aoi_casual_ending" in checks


def test_duplicate_detected():
    r = _check("case01_coffee", [("ren", "今日はゆっくり休みましょう。"), ("aoi", "今日はゆっくり休みましょうね。")])
    assert any(f.check == "no_duplicate_utterance" for f in r.failures)


def test_case03_ren_forbidden():
    r = _check("case03_ren_praise", [("ren", "俺、見てたよ。よく作ったな")])
    assert {f.detail for f in r.failures} >= {"俺", "よく作ったな"}
    ok = _check("case03_ren_praise", [("ren", "……頑張ったね。よく作ったね")])
    assert ok.passed


def test_case05_no_acknowledgement():
    r = _check("case05_correction_aoi", [("aoi", "承知しました。では続きです。")])
    assert not r.passed
    ok = _check("case05_correction_aoi", [("aoi", "それで、さっきの散歩の話ですが、あの坂は少し急でしたね。")])
    assert ok.passed


def test_case08_short_and_polite():
    r = _check("case08_hurry_short", [("aoi", "了解、すぐやろう")])
    assert not r.passed
    ok = _check("case08_hurry_short", [("aoi", "はい。結論から伝えます。")])
    assert ok.passed


def test_unknown_speaker_and_narration():
    r = _check("case10_tech_worry", [("narrator", "二人は顔を見合わせた"), ("ren", "（うなずいて）……うん")])
    checks = {f.check for f in r.failures}
    assert {"allowed_speakers", "no_narration"} <= checks


def test_ren_command():
    r = _check("case07_just_talk", [("ren", "早く寝ろ")])
    assert any(f.check == "ren_command_ending" for f in r.failures)
