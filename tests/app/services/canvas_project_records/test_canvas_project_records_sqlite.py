"""Tests for SqliteCanvasProjectRecordStorage.

Covers the CRUD + filter operations, the in-place metadata update used by the
`PUT /i/{name}/file` endpoint, and the multiuser isolation contract that pins
the behaviour for `get_many` / `get_project_names` when ``board_id`` is omitted.
"""

import pytest

from invokeai.app.services.canvas_project_records.canvas_project_records_common import (
    CanvasProjectRecordChanges,
    CanvasProjectRecordNotFoundException,
)
from invokeai.app.services.canvas_project_records.canvas_project_records_sqlite import (
    SqliteCanvasProjectRecordStorage,
)
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def store() -> SqliteCanvasProjectRecordStorage:
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return SqliteCanvasProjectRecordStorage(db=db)


def _save(
    store: SqliteCanvasProjectRecordStorage,
    project_name: str,
    user_id: str = "system",
    name: str = "Test Project",
    has_thumbnail: bool = False,
    starred: bool = False,
) -> None:
    store.save(
        project_name=project_name,
        project_origin=ResourceOrigin.INTERNAL,
        name=name,
        app_version="test-1.0",
        width=1024,
        height=1024,
        image_count=2,
        has_thumbnail=has_thumbnail,
        starred=starred,
        user_id=user_id,
    )


class TestCrud:
    def test_save_then_get_round_trips_fields(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", user_id="alice", name="My Project", has_thumbnail=True, starred=True)
        record = store.get("p1")
        assert record.project_name == "p1"
        assert record.name == "My Project"
        assert record.app_version == "test-1.0"
        assert record.width == 1024
        assert record.height == 1024
        assert record.image_count == 2
        assert record.has_thumbnail is True
        assert record.starred is True
        assert record.user_id == "alice"
        assert record.project_origin == ResourceOrigin.INTERNAL

    def test_get_missing_raises(self, store: SqliteCanvasProjectRecordStorage) -> None:
        with pytest.raises(CanvasProjectRecordNotFoundException):
            store.get("does-not-exist")

    def test_delete_removes_record(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1")
        store.delete("p1")
        with pytest.raises(CanvasProjectRecordNotFoundException):
            store.get("p1")

    def test_delete_many(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1")
        _save(store, "p2")
        _save(store, "p3")
        store.delete_many(["p1", "p2"])
        # The remaining one survived.
        assert store.get("p3").project_name == "p3"
        with pytest.raises(CanvasProjectRecordNotFoundException):
            store.get("p1")
        with pytest.raises(CanvasProjectRecordNotFoundException):
            store.get("p2")

    def test_get_user_id_returns_none_when_missing(self, store: SqliteCanvasProjectRecordStorage) -> None:
        assert store.get_user_id("missing") is None

    def test_get_user_id_returns_owner(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", user_id="bob")
        assert store.get_user_id("p1") == "bob"


class TestUpdate:
    def test_update_name(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", name="Original")
        store.update("p1", CanvasProjectRecordChanges(name="Renamed"))
        assert store.get("p1").name == "Renamed"

    def test_update_starred(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", starred=False)
        store.update("p1", CanvasProjectRecordChanges(starred=True))
        assert store.get("p1").starred is True

    def test_update_is_intermediate(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1")
        store.update("p1", CanvasProjectRecordChanges(is_intermediate=True))
        assert store.get("p1").is_intermediate is True

    def test_update_with_no_changes_is_noop(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", name="Keep")
        store.update("p1", CanvasProjectRecordChanges())
        assert store.get("p1").name == "Keep"


class TestSetHasThumbnail:
    def test_toggle_has_thumbnail(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", has_thumbnail=False)
        store.set_has_thumbnail("p1", True)
        assert store.get("p1").has_thumbnail is True
        store.set_has_thumbnail("p1", False)
        assert store.get("p1").has_thumbnail is False


class TestUpdateFileMetadata:
    """`update_file_metadata` is called by `PUT /i/{name}/file` when the ZIP is replaced in
    place. It must atomically refresh dimensions, image_count, thumbnail flag and app_version
    while preserving everything else (board, starred, ownership)."""

    def test_updates_all_file_fields(self, store: SqliteCanvasProjectRecordStorage) -> None:
        _save(store, "p1", name="Keep", has_thumbnail=False, starred=True)
        # Sanity: starred is set from the seed.
        assert store.get("p1").starred is True

        store.update_file_metadata(
            project_name="p1",
            width=2048,
            height=1024,
            image_count=7,
            has_thumbnail=True,
            app_version="test-1.1",
        )

        record = store.get("p1")
        assert record.width == 2048
        assert record.height == 1024
        assert record.image_count == 7
        assert record.has_thumbnail is True
        assert record.app_version == "test-1.1"
        # Things we must NOT clobber on a file-replace:
        assert record.name == "Keep"
        assert record.starred is True
        assert record.user_id == "system"


class TestGetManyMultiuserIsolation:
    """When `board_id` is omitted, non-admin callers must only see their own projects.

    Mirrors the regression test added for videos in PR #9163 / test_video_records_sqlite.py.
    """

    @pytest.fixture
    def seeded(self, store: SqliteCanvasProjectRecordStorage) -> SqliteCanvasProjectRecordStorage:
        _save(store, "alice_1", user_id="alice")
        _save(store, "alice_2", user_id="alice")
        _save(store, "bob_1", user_id="bob")
        _save(store, "bob_2", user_id="bob")
        return store

    def test_non_admin_only_sees_own_projects(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        result = seeded.get_many(user_id="alice", is_admin=False)
        names = {p.project_name for p in result.items}
        assert names == {"alice_1", "alice_2"}
        assert result.total == 2

    def test_admin_sees_every_users_projects(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        result = seeded.get_many(user_id="alice", is_admin=True)
        names = {p.project_name for p in result.items}
        assert names == {"alice_1", "alice_2", "bob_1", "bob_2"}

    def test_no_user_id_returns_all(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        result = seeded.get_many(user_id=None, is_admin=False)
        names = {p.project_name for p in result.items}
        assert names == {"alice_1", "alice_2", "bob_1", "bob_2"}

    def test_explicit_none_board_still_isolates(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        # "none" sentinel = uncategorized. Must still enforce per-user filtering.
        result = seeded.get_many(board_id="none", user_id="alice", is_admin=False)
        names = {p.project_name for p in result.items}
        assert names == {"alice_1", "alice_2"}


class TestGetProjectNamesMultiuserIsolation:
    @pytest.fixture
    def seeded(self, store: SqliteCanvasProjectRecordStorage) -> SqliteCanvasProjectRecordStorage:
        _save(store, "alice_1", user_id="alice")
        _save(store, "alice_2", user_id="alice")
        _save(store, "bob_1", user_id="bob")
        return store

    def test_non_admin_only_sees_own(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        result = seeded.get_project_names(user_id="alice", is_admin=False)
        assert set(result.project_names) == {"alice_1", "alice_2"}
        assert result.total_count == 2

    def test_admin_sees_all(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        result = seeded.get_project_names(user_id="alice", is_admin=True)
        assert set(result.project_names) == {"alice_1", "alice_2", "bob_1"}


class TestGetManyFilters:
    @pytest.fixture
    def seeded(self, store: SqliteCanvasProjectRecordStorage) -> SqliteCanvasProjectRecordStorage:
        _save(store, "p_starred", starred=True, name="Starred One")
        _save(store, "p_normal", starred=False, name="Normal Two")
        _save(store, "p_intermediate", starred=False)
        store.update("p_intermediate", CanvasProjectRecordChanges(is_intermediate=True))
        return store

    def test_starred_first_orders_starred_above_normal(
        self, seeded: SqliteCanvasProjectRecordStorage
    ) -> None:
        # Two non-intermediate projects: one starred, one not. Starred-first ordering
        # should put the starred one first.
        result = seeded.get_many(starred_first=True, is_intermediate=False)
        starred_names = [p.project_name for p in result.items if p.starred]
        normal_names = [p.project_name for p in result.items if not p.starred]
        assert starred_names == ["p_starred"]
        assert normal_names == ["p_normal"]
        # The starred project comes first in the result list.
        assert result.items[0].project_name == "p_starred"

    def test_is_intermediate_filter(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        all_projects = seeded.get_many(is_intermediate=None)
        non_intermediate = seeded.get_many(is_intermediate=False)
        intermediate_only = seeded.get_many(is_intermediate=True)
        assert all_projects.total == 3
        assert non_intermediate.total == 2
        assert intermediate_only.total == 1
        assert intermediate_only.items[0].project_name == "p_intermediate"

    def test_search_term_matches_name(self, seeded: SqliteCanvasProjectRecordStorage) -> None:
        result = seeded.get_many(search_term="Starred")
        assert {p.project_name for p in result.items} == {"p_starred"}
