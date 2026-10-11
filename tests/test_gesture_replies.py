import json
from kodama.reply import parse_reply, reply_schema
from kodama.config import Config
from kodama.conversation import ConversationService
from kodama.model.mock import ScriptedAdapter, MockAdapter
from kodama.storage.sqlite import SQLiteStore
from seed import PACK, CAST


def test_optional_safe_ids_and_per_speaker_capabilities():
    offered = {"ren": ["nod", "wave", "file:///x"], "aoi": ["bow"]}
    values = [None, "nod", {"path": "/tmp/x"}, "raiseHand", 12, "bow"]
    for v in values:
        raw = json.dumps({"utterances": [{"speaker": "ren", "text": "はい。", "gesture": v}]})
        parsed = parse_reply(raw, CAST.character_ids, offered)
        assert parsed.utterances == [("ren", "はい。")]
        assert parsed.gestures == (["nod"] if v == "nod" else ["none"])
    assert parse_reply('{"utterances":[{"speaker":"ren","text":"旧返答"}]}', CAST.character_ids, offered).gestures == ["none"]
    enum = reply_schema(CAST.character_ids, offered)["properties"]["utterances"]["items"]["properties"]["gesture"]["enum"]
    assert enum == ["none", "nod", "wave", "bow"]


def test_service_attaches_ids_without_changing_message_storage(tmp_path):
    store = SQLiteStore(str(tmp_path / "isolated.db"), cast=CAST)
    adapter = ScriptedAdapter([{"utterances": [{"speaker": "ren", "text": "うん。", "gesture": "nod"}, {"speaker": "ren", "text": "続き。", "gesture": "bow"}], "memory_candidates": []}])
    service = ConversationService(store, adapter, Config(memory_auto_approve=False), PACK, gesture_capabilities={"ren": ["nod"]})
    service.startup()
    session = store.create_session("gesture mock test")
    result = service.send(session.id, "テスト")
    assert result.ok and result.reply_gestures == {result.replies[0].id: "nod", result.replies[1].id: "none"}
    assert "gesture" not in store.list_messages(session.id)[1].__dict__
    request = adapter.requests[0]
    assert request.metadata["gesture_capabilities"] == {"ren": ["nod"], "aoi": []}
    assert "毎回動かさず" in request.system_blocks[-1]
    assert "memory_candidates" in request.output_schema["properties"]
    store.close()


def test_mock_greeting_and_ordinary_none(tmp_path):
    store = SQLiteStore(str(tmp_path / "mock.db"), cast=CAST)
    service = ConversationService(store, MockAdapter(), Config(memory_auto_approve=False), PACK, gesture_capabilities={"ren": ["wave"], "aoi": ["wave"]})
    service.startup(); session = store.create_session("mock")
    assert set(service.send(session.id, "こんにちは、ふたりとも").reply_gestures.values()) == {"wave"}
    assert set(service.send(session.id, "今日は静かに過ごした").reply_gestures.values()) == {"none"}
    store.close()


def test_mock_none_for_sad_apology_or_long_explanation():
    from kodama.model.mock import choose_gesture
    for text in ["今日は悲しい。ありがとう", "ごめんね、こんにちは", "長い説明"*30]:
        assert choose_gesture(text) == "none"
