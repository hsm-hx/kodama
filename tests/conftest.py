from __future__ import annotations

import pytest

from kodama import migration
from kodama.storage.sqlite import SQLiteStore
from seed import seed_store


@pytest.fixture(params=["sqlite", "sqlite_after_roundtrip"])
def seeded(request, tmp_path):
    """(store, SeedIds). The second variant is the seed exported and imported into a fresh DB,
    so every contract assertion also checks that meaning survives a migration."""
    source = SQLiteStore(tmp_path / "source.db")
    ids = seed_store(source)
    if request.param == "sqlite":
        yield source, ids
        source.close()
        return
    export_path = tmp_path / "export.json"
    migration.export_to_file(source, export_path)
    source.close()
    target = tmp_path / "migrated.db"
    migration.import_file(export_path, target, active_db_path=tmp_path / "source.db")
    store = SQLiteStore(target)
    yield store, ids
    store.close()


@pytest.fixture
def store(seeded):
    return seeded[0]


@pytest.fixture
def seed(seeded):
    return seeded[1]


@pytest.fixture
def empty_store(tmp_path):
    s = SQLiteStore(tmp_path / "empty.db")
    yield s
    s.close()
