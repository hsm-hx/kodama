"""Grouped updates must be all-or-nothing, and retries must not double-record."""

from __future__ import annotations

import pytest

from kodama import migration
from kodama.domain import MemoryDraft, MemoryKind, MemoryOrigin, MemoryStatus, NodeType, TurnStatus
from kodama.storage.sqlite import SQLiteStore
from seed import seed_store


class Boom(Exception):
    pass


class Fault:
    def __init__(self):
        self.point: str | None = None

    def __call__(self, point: str) -> None:
        if point == self.point:
            self.point = None  # fire once
            raise Boom(point)


@pytest.fixture
def faulty(tmp_path):
    fault = Fault()
    store = SQLiteStore(tmp_path / "f.db", fault_hook=fault)
    yield store, fault
    store.close()


@pytest.mark.parametrize("point", ["complete_turn.after_messages", "complete_turn.after_candidates"])
def test_complete_turn_is_atomic_and_retry_safe(faulty, point):
    store, fault = faulty
    session = store.create_session("s")
    turn, _ = store.begin_turn(session.id, "やあ", "mock", "m")
    cand = [MemoryDraft(body="候補", kind=MemoryKind.USER_STATED)]
    fault.point = point
    with pytest.raises(Boom):
        store.complete_turn(turn.id, [("ren", "うん"), ("aoi", "はい")], candidates=cand)
    assert store.get_turn(turn.id).status == TurnStatus.PENDING
    assert len(store.list_messages(session.id)) == 1
    assert store.list_memory_versions() == []

    first = store.complete_turn(turn.id, [("ren", "うん"), ("aoi", "はい")], candidates=cand)
    second = store.complete_turn(turn.id, [("ren", "別の返答")], candidates=cand)
    assert [m.id for m in second] == [m.id for m in first]
    assert len(store.list_messages(session.id)) == 3
    assert len(store.list_memory_versions()) == 1
    assert store.get_turn(turn.id).status == TurnStatus.COMPLETED


@pytest.mark.parametrize("point", ["revise_memory.after_insert", "revise_memory.after_supersede"])
def test_revise_memory_is_atomic_and_retry_safe(faulty, point):
    store, fault = faulty
    old = store.add_memory(
        MemoryDraft(body="旧", kind=MemoryKind.USER_STATED), MemoryOrigin.USER_EXPLICIT, MemoryStatus.APPROVED
    )
    draft = MemoryDraft(body="新", kind=MemoryKind.USER_STATED)
    fault.point = point
    with pytest.raises(Boom):
        store.revise_memory(old.id, draft, "訂正")
    assert [v.id for v in store.list_memory_versions()] == [old.id]
    assert store.get_memory_version(old.id).status == MemoryStatus.APPROVED
    assert store.links_of(NodeType.MEMORY_VERSION, old.id) == []

    new = store.revise_memory(old.id, draft, "訂正")
    assert len(store.list_memory_versions()) == 2
    assert len(store.links_of(NodeType.MEMORY_VERSION, old.id)) == 1
    # retrying the same correction does not create a second replacement
    with pytest.raises(Exception):
        store.revise_memory(old.id, draft, "訂正")
    assert [v.id for v in store.list_memory_versions([MemoryStatus.APPROVED])] == [new.id]


@pytest.mark.parametrize("point", ["import_snapshot.after_messages", "import_snapshot.after_links"])
def test_import_snapshot_is_atomic_and_retry_safe(tmp_path, point):
    src = SQLiteStore(tmp_path / "src.db")
    seed_store(src)
    data = src.export_snapshot()
    src.close()

    fault = Fault()
    target = SQLiteStore(tmp_path / "dst.db", fault_hook=fault)
    fault.point = point
    with pytest.raises(Boom):
        target.import_snapshot(data)
    assert all(v == [] for v in target.export_snapshot().values())

    stats = target.import_snapshot(data)
    assert sum(stats.inserted.values()) == sum(len(v) for v in data.values())
    again = target.import_snapshot(data)
    assert sum(again.inserted.values()) == 0
    assert target.export_snapshot() == data
    target.close()


def test_recover_marks_pending_as_interrupted_without_resend(tmp_path):
    path = tmp_path / "r.db"
    store = SQLiteStore(path)
    session = store.create_session("s")
    turn, _ = store.begin_turn(session.id, "送信中に落ちる", "mock", "m")
    store.close()  # simulated crash while the API call was in flight

    reopened = SQLiteStore(path)
    recovered = reopened.recover_incomplete_turns()
    assert [t.id for t in recovered] == [turn.id]
    assert recovered[0].status == TurnStatus.INTERRUPTED
    assert reopened.recover_incomplete_turns() == []
    msgs = reopened.list_messages(session.id)
    assert [m.text for m in msgs] == ["送信中に落ちる"]  # input kept, no reply invented
    # a new turn can start; the interrupted one is not resent
    reopened.begin_turn(session.id, "次の入力", "mock", "m")
    assert len(reopened.list_turns(session.id)) == 2
    reopened.close()


def test_keyboard_interrupt_rolls_back(tmp_path):
    def hook(point):
        if point == "complete_turn.after_messages":
            raise KeyboardInterrupt

    store = SQLiteStore(tmp_path / "k.db", fault_hook=hook)
    session = store.create_session("s")
    turn, _ = store.begin_turn(session.id, "やあ", "mock", "m")
    with pytest.raises(KeyboardInterrupt):
        store.complete_turn(turn.id, [("ren", "うん")])
    store.fault_hook = None
    assert len(store.list_messages(session.id)) == 1
    store.fail_turn(turn.id, TurnStatus.INTERRUPTED, "Ctrl+C")
    assert store.get_turn(turn.id).status == TurnStatus.INTERRUPTED
    store.close()


def test_import_file_failure_leaves_no_target(tmp_path, monkeypatch):
    src = SQLiteStore(tmp_path / "src.db")
    seed_store(src)
    export = tmp_path / "e.json"
    migration.export_to_file(src, export)
    src.close()

    original = SQLiteStore.import_snapshot

    def failing(self, data):
        if "importing" in self.path:  # only the real target, not the verification scratch DB

            def hook(point):
                if point.endswith("memory_versions"):
                    raise Boom(point)

            self.fault_hook = hook
        return original(self, data)

    monkeypatch.setattr(SQLiteStore, "import_snapshot", failing)
    target = tmp_path / "new.db"
    with pytest.raises(Boom):
        migration.import_file(export, target)
    assert not target.exists()
    assert [p.name for p in tmp_path.iterdir() if "importing" in p.name] == []


@pytest.mark.parametrize("point", ["complete_turn.after_messages", "complete_turn.after_candidates"])
def test_complete_turn_with_auto_approved_memory_is_atomic(faulty, point):
    store, fault = faulty
    session = store.create_session("s")
    turn, _ = store.begin_turn(session.id, "やあ", "mock", "m")
    cand = [MemoryDraft(body="記憶", kind=MemoryKind.USER_STATED)]
    fault.point = point
    with pytest.raises(Boom):
        store.complete_turn(turn.id, [("ren", "うん")], candidates=cand, candidate_status=MemoryStatus.APPROVED)
    assert store.get_turn(turn.id).status == TurnStatus.PENDING
    assert len(store.list_messages(session.id)) == 1
    assert store.list_memory_versions() == []
    store.complete_turn(turn.id, [("ren", "うん")], candidates=cand, candidate_status=MemoryStatus.APPROVED)
    [v] = store.list_memory_versions()
    assert v.status == MemoryStatus.APPROVED and v.origin == MemoryOrigin.MODEL_CANDIDATE
