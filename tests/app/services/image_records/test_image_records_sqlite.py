"""DB-backed tests for SqliteImageRecordStorage.

Verifies that image_subfolder round-trips correctly through save(), get(),
get_many(), and get_intermediates() against a real (in-memory) SQLite database,
and that get_many()/get_image_names() enforce per-user ownership isolation.
"""

import sqlite3

import pytest

from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordChanges,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def store() -> SqliteImageRecordStorage:
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return SqliteImageRecordStorage(db=db)


@pytest.fixture
def stores() -> tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage]:
    """Image, board, and board-image storages sharing one in-memory database."""
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return (
        SqliteImageRecordStorage(db=db),
        SqliteBoardRecordStorage(db=db),
        SqliteBoardImageRecordStorage(db=db),
    )


def _save(
    store: SqliteImageRecordStorage,
    name: str,
    subfolder: str = "",
    is_intermediate: bool = False,
    user_id: str | None = None,
    category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    store.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=is_intermediate,
        image_subfolder=subfolder,
        user_id=user_id,
    )


def _capture_names_plan(store: SqliteImageRecordStorage, **kwargs):
    statements: list[str] = []
    store._db._conn.set_trace_callback(statements.append)
    try:
        result = store.get_image_names(**kwargs)
    finally:
        store._db._conn.set_trace_callback(None)
    statement = next(statement for statement in statements if "SELECT images.image_name" in statement)
    details = [row[3] for row in store._db._conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()]
    return result, statement, details


class TestImageSubfolderRoundTrip:
    """save() -> get() preserves image_subfolder."""

    def test_default_empty_subfolder(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "img_default.png")
        record = store.get("img_default.png")
        assert record.image_subfolder == ""

    def test_custom_subfolder(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "img_sub.png", subfolder="2026/04/11")
        record = store.get("img_sub.png")
        assert record.image_subfolder == "2026/04/11"

    def test_nested_subfolder(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "img_nested.png", subfolder="a/b/c/d")
        record = store.get("img_nested.png")
        assert record.image_subfolder == "a/b/c/d"


class TestGetManySubfolder:
    """get_many() deserializes image_subfolder for every row."""

    def test_get_many_returns_subfolders(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "flat.png", subfolder="")
        _save(store, "dated.png", subfolder="2026/01")
        _save(store, "hashed.png", subfolder="ab")

        result = store.get_many(limit=10, order_dir=SQLiteDirection.Ascending)
        by_name = {r.image_name: r.image_subfolder for r in result.items}

        assert by_name["flat.png"] == ""
        assert by_name["dated.png"] == "2026/01"
        assert by_name["hashed.png"] == "ab"


class TestGetIntermediatesSubfolder:
    """get_intermediates() returns (name, subfolder) pairs without deleting rows."""

    def test_returns_subfolder_pairs(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "keep.png", subfolder="general", is_intermediate=False)
        _save(store, "tmp1.png", subfolder="intermediate", is_intermediate=True)
        _save(store, "tmp2.png", subfolder="intermediate", is_intermediate=True)

        pairs = store.get_intermediates()

        # Should return only intermediate images with their subfolders
        assert len(pairs) == 2
        names_and_subs = set(pairs)
        assert ("tmp1.png", "intermediate") in names_and_subs
        assert ("tmp2.png", "intermediate") in names_and_subs

        # Non-intermediate image should still exist
        record = store.get("keep.png")
        assert record.image_subfolder == "general"

    def test_get_intermediates_does_not_delete(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        store.get_intermediates()

        # Listing intermediates must not remove them.
        record = store.get("tmp.png")
        assert record.image_subfolder == "x"

    def test_intermediates_are_deleted_via_delete_intermediates_by_names(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        pairs = store.get_intermediates()
        deleted = store.delete_intermediates_by_names([name for name, _ in pairs])

        assert deleted == ["tmp.png"]
        with pytest.raises(ImageRecordNotFoundException):
            store.get("tmp.png")


class TestQueryFaultsAreNotNotFound:
    """A failing query means the database is unavailable, not that the image is missing.

    Reporting a query fault as "not found" propagates all the way to the API, where it becomes a 404
    and tells the frontend to drop a live image from its cache.
    """

    def _break_the_images_table(self, store: SqliteImageRecordStorage) -> None:
        store._db._conn.execute("ALTER TABLE images RENAME TO images_moved;")

    def test_get_raises_the_db_error_not_not_found(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "live.png")
        self._break_the_images_table(store)

        with pytest.raises(sqlite3.Error):
            store.get("live.png")

    def test_get_metadata_raises_the_db_error_not_not_found(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "live.png")
        self._break_the_images_table(store)

        with pytest.raises(sqlite3.Error):
            store.get_metadata("live.png")

    def test_missing_row_still_raises_not_found(self, store: SqliteImageRecordStorage) -> None:
        """The genuine not-found path is untouched."""
        with pytest.raises(ImageRecordNotFoundException):
            store.get("never-existed.png")
        with pytest.raises(ImageRecordNotFoundException):
            store.get_metadata("never-existed.png")


class TestDeleteIntermediatesByNames:
    """delete_intermediates_by_names() deletes only rows that are still intermediates."""

    def test_promoted_image_keeps_its_record(self, store: SqliteImageRecordStorage) -> None:
        """An image promoted out of intermediate status after the snapshot must survive."""
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        _save(store, "promoted.png", subfolder="x", is_intermediate=True)
        snapshot = [name for name, _ in store.get_intermediates()]
        assert set(snapshot) == {"tmp.png", "promoted.png"}

        # Simulate the race: the image stops being an intermediate between the snapshot and delete.
        store.update("promoted.png", ImageRecordChanges(is_intermediate=False))

        deleted = store.delete_intermediates_by_names(snapshot)

        assert deleted == ["tmp.png"]
        # promoted.png is excluded from the returned names, so the caller never purges its files.
        assert store.get("promoted.png").is_intermediate is False
        with pytest.raises(ImageRecordNotFoundException):
            store.get("tmp.png")

    def test_promotion_interleaved_inside_the_call_keeps_the_record(self, store: SqliteImageRecordStorage) -> None:
        """The is_intermediate predicate must ride on the DELETE, not on a preceding SELECT.

        Python's legacy sqlite3 transaction control opens a transaction only before a write, so a
        SELECT inside this method holds no read lock. A writer that promotes an image after that
        SELECT but before the DELETE must still not lose its record.
        """
        _save(store, "tmp.png", is_intermediate=True)
        _save(store, "promoted.png", is_intermediate=True)
        snapshot = [name for name, _ in store.get_intermediates()]

        # Promote from inside the call, between the first SELECT and the DELETE.
        real_execute = store._db._conn.execute
        promoted = False

        def trace(statement: str) -> None:
            nonlocal promoted
            # The trace fires when a statement *begins*, so hooking the first SELECT would promote
            # before that SELECT reads anything — indistinguishable from promoting up front. Hooking
            # the DELETE puts the promotion after the SELECT has already seen the row as an
            # intermediate, which is the interleaving that a SELECT-then-unconditional-DELETE
            # implementation gets wrong.
            if not promoted and statement.strip().upper().startswith("DELETE FROM IMAGES"):
                promoted = True
                real_execute("UPDATE images SET is_intermediate = 0 WHERE image_name = 'promoted.png'")

        store._db._conn.set_trace_callback(trace)
        try:
            deleted = store.delete_intermediates_by_names(snapshot)
        finally:
            store._db._conn.set_trace_callback(None)

        assert promoted, "the interleaved promotion never ran; the test proves nothing"
        assert deleted == ["tmp.png"]
        assert store.get("promoted.png").is_intermediate is False

    def test_unknown_and_empty_names_are_no_ops(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "keep.png", is_intermediate=False)

        assert store.delete_intermediates_by_names([]) == []
        # "gone.png" has no record at all and "keep.png" is not an intermediate, so neither is
        # deleted or returned; keep.png must still be present afterwards.
        assert store.delete_intermediates_by_names(["gone.png", "keep.png"]) == []
        assert store.get("keep.png").image_name == "keep.png"

    def test_more_names_than_sql_variable_limit(self, store: SqliteImageRecordStorage) -> None:
        """Chunking must not lose rows: exercise a name list spanning several chunks."""
        chunk = SqliteImageRecordStorage._MAX_SQL_VARIABLES
        names = [f"tmp{i:05d}.png" for i in range(chunk * 2 + 7)]
        for name in names:
            _save(store, name, is_intermediate=True)
        # One image in the middle chunk is promoted and must survive.
        survivor = names[chunk + 3]
        store.update(survivor, ImageRecordChanges(is_intermediate=False))

        deleted = store.delete_intermediates_by_names(names)

        assert set(deleted) == set(names) - {survivor}
        assert survivor not in deleted
        assert store.get(survivor).is_intermediate is False
        assert store.get_intermediates() == []

    def test_chunking_stays_within_the_declared_variable_limit(self, store: SqliteImageRecordStorage) -> None:
        """No statement may bind more parameters than the declared limit."""
        chunk = SqliteImageRecordStorage._MAX_SQL_VARIABLES
        names = [f"tmp{i:05d}.png" for i in range(chunk * 2 + 7)]
        for name in names:
            _save(store, name, is_intermediate=True)

        # The trace callback reports statements with their parameters already expanded, so count the
        # bound image names in each one rather than the placeholders.
        widest = 0

        def trace(statement: str) -> None:
            nonlocal widest
            if "images WHERE image_name IN (" in statement:
                widest = max(widest, statement.count(".png"))

        store._db._conn.set_trace_callback(trace)
        try:
            store.delete_intermediates_by_names(names)
        finally:
            store._db._conn.set_trace_callback(None)

        # 999 is the SQLITE_MAX_VARIABLE_NUMBER default on builds older than 3.32. Asserting the
        # literal rather than _MAX_SQL_VARIABLES keeps the test meaningful if that constant is raised.
        assert 0 < widest <= 999


class TestOwnershipFilteringOmittedBoard:
    """get_many()/get_image_names() enforce per-user isolation when board_id is omitted.

    Without this, a non-admin could enumerate every user's images (including images
    on other users' private boards) simply by omitting the board_id query parameter.
    """

    def _seed_two_users(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> str:
        """user1: one image on a private board + one uncategorized. user2: one uncategorized."""
        image_store, board_store, board_image_store = stores
        _save(image_store, "u1-boarded.png", user_id="user1")
        _save(image_store, "u1-uncat.png", user_id="user1")
        _save(image_store, "u2-uncat.png", user_id="user2")
        board = board_store.save(board_name="User1 Private Board", user_id="user1")
        board_image_store.add_image_to_board(board_id=board.board_id, image_name="u1-boarded.png")
        return board.board_id

    def test_get_many_omitted_board_filters_by_owner(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id="user2", is_admin=False)

        assert {r.image_name for r in result.items} == {"u2-uncat.png"}
        assert result.total == 1

    def test_get_many_omitted_board_owner_sees_boarded_and_uncategorized(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id="user1", is_admin=False)

        assert {r.image_name for r in result.items} == {"u1-boarded.png", "u1-uncat.png"}
        assert result.total == 2

    def test_get_many_omitted_board_admin_sees_all(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id="admin", is_admin=True)

        assert {r.image_name for r in result.items} == {"u1-boarded.png", "u1-uncat.png", "u2-uncat.png"}
        assert result.total == 3

    def test_get_many_omitted_board_single_user_mode_sees_all(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        """user_id=None (single-user mode) applies no ownership filter."""
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id=None, is_admin=False)

        assert result.total == 3

    def test_get_many_none_board_still_filters_by_owner(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        """board_id="none" (uncategorized) keeps its existing per-user isolation."""
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, board_id="none", user_id="user1", is_admin=False)

        assert {r.image_name for r in result.items} == {"u1-uncat.png"}

    def test_get_many_explicit_board_returns_board_contents(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        """An explicit board_id lists that board's images; read access is the router's job."""
        board_id = self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, board_id=board_id, user_id="user1", is_admin=False)

        assert {r.image_name for r in result.items} == {"u1-boarded.png"}

    def test_get_image_names_omitted_board_filters_by_owner(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(user_id="user2", is_admin=False)

        assert result.image_names == ["u2-uncat.png"]
        assert result.total_count == 1

    def test_get_image_names_omitted_board_admin_sees_all(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(user_id="admin", is_admin=True)

        assert set(result.image_names) == {"u1-boarded.png", "u1-uncat.png", "u2-uncat.png"}
        assert result.total_count == 3

    def test_get_image_names_omitted_board_single_user_mode_sees_all(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(user_id=None, is_admin=False)

        assert result.total_count == 3

    def test_get_image_names_none_board_still_filters_by_owner(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(board_id="none", user_id="user1", is_admin=False)

        assert result.image_names == ["u1-uncat.png"]


class TestGetImageNamesQueryPlans:
    def test_default_category_query_has_no_forced_index(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "image.png", user_id="alice")

        result, statement, _ = _capture_names_plan(
            store,
            categories=[ImageCategory.GENERAL],
            is_intermediate=False,
            is_admin=True,
        )

        assert result.image_names == ["image.png"]
        assert "LEFT JOIN board_images" not in statement
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"search_term": "does-not-match"},
            {"user_id": "alice", "is_admin": False},
            {"user_id": "alice", "is_admin": False, "starred_first": False},
            {"user_id": "alice", "is_admin": False, "order_dir": SQLiteDirection.Ascending},
        ],
    )
    def test_query_shapes_do_not_force_indexes(
        self,
        store: SqliteImageRecordStorage,
        kwargs,
    ) -> None:
        _save(store, "image.png", user_id="alice")

        _, statement, _ = _capture_names_plan(
            store,
            categories=[ImageCategory.GENERAL],
            is_intermediate=False,
            **kwargs,
        )

        assert "LEFT JOIN board_images" not in statement
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    def test_none_board_uses_anti_membership_filter(self, stores) -> None:
        image_store, board_store, board_image_store = stores
        _save(image_store, "boarded.png", user_id="alice")
        _save(image_store, "unboarded.png", user_id="alice")
        board = board_store.save("Board", "alice")
        board_image_store.add_image_to_board(board.board_id, "boarded.png")

        result, statement, _ = _capture_names_plan(
            image_store,
            board_id="none",
            user_id="alice",
            is_admin=False,
            starred_first=False,
        )

        assert result.image_names == ["unboarded.png"]
        assert "NOT EXISTS" in statement
        assert "LEFT JOIN board_images" not in statement

    def test_nonadmin_asset_query_has_no_forced_index(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "asset.png", user_id="alice", category=ImageCategory.CONTROL)

        result, statement, _ = _capture_names_plan(
            store,
            categories=[ImageCategory.CONTROL],
            is_intermediate=False,
            user_id="alice",
            is_admin=False,
        )

        assert result.image_names == ["asset.png"]
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement
