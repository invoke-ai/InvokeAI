"""DB-backed tests for SqliteImageRecordStorage.

Verifies that image_subfolder round-trips correctly through save(), get(),
get_many(), and get_subfolders() against a real (in-memory) SQLite database,
and that get_many()/get_image_names() enforce per-user ownership isolation.
"""

import sqlite3

import pytest

from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility
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


class TestImageRecordExists:
    def test_exists_returns_true_for_saved_image(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "exists.png")

        assert store.exists("exists.png") is True

    def test_exists_returns_false_for_missing_image(self, store: SqliteImageRecordStorage) -> None:
        assert store.exists("missing.png") is False


def _intermediates(store: SqliteImageRecordStorage) -> list[tuple[str, str]]:
    with store._db.transaction() as cursor:
        cursor.execute("SELECT image_name, image_subfolder FROM images WHERE is_intermediate = TRUE;")
        return [(row[0], row[1]) for row in cursor.fetchall()]


class TestGetSubfolders:
    """get_subfolders() maps the named rows to their on-disk subfolders without touching them."""

    def test_returns_subfolders_of_existing_rows_only(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "keep.png", subfolder="general", is_intermediate=False)
        _save(store, "tmp1.png", subfolder="intermediate", is_intermediate=True)

        assert store.get_subfolders(["keep.png", "tmp1.png", "missing.png"]) == {
            "keep.png": "general",
            "tmp1.png": "intermediate",
        }
        assert store.get("tmp1.png").image_subfolder == "intermediate"

    def test_intermediates_are_deleted_via_delete_intermediates_by_names(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        deleted = store.delete_intermediates_by_names([name for name, _ in _intermediates(store)])

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
        snapshot = [name for name, _ in _intermediates(store)]
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
        snapshot = [name for name, _ in _intermediates(store)]

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
        assert _intermediates(store) == []

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


class TestAllReadableBoardsFiltering:
    def _seed_visibility_matrix(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        image_store, board_store, board_image_store = stores

        def add_boarded_image(
            image_name: str,
            *,
            owner: str,
            visibility: BoardVisibility = BoardVisibility.Private,
            archived: bool = False,
        ) -> str:
            _save(image_store, image_name, user_id=owner)
            board = board_store.save(board_name=image_name, user_id=owner)
            board_store.update(
                board.board_id,
                BoardChanges(board_visibility=visibility, archived=archived),
            )
            board_image_store.add_image_to_board(board_id=board.board_id, image_name=image_name)
            return board.board_id

        add_boarded_image("own-private.png", owner="user1")
        shared_private_id = add_boarded_image("explicit-share.png", owner="user2")
        add_boarded_image("shared-visibility.png", owner="user2", visibility=BoardVisibility.Shared)
        add_boarded_image("public-visibility.png", owner="user2", visibility=BoardVisibility.Public)
        add_boarded_image("other-private.png", owner="user2")
        add_boarded_image("own-archived.png", owner="user1", archived=True)
        add_boarded_image("other-archived-public.png", owner="user2", visibility=BoardVisibility.Public, archived=True)
        _save(image_store, "own-uncategorized.png", user_id="user1")
        _save(image_store, "other-uncategorized.png", user_id="user2")

        with image_store._db.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, email, password_hash) VALUES (?, ?, ?)",
                ("user1", "user1@example.com", "unused"),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO shared_boards (board_id, user_id) VALUES (?, ?)",
                (shared_private_id, "user1"),
            )

    @pytest.mark.parametrize(
        ("user_id", "is_admin", "expected"),
        [
            (
                "user1",
                False,
                {
                    "own-private.png",
                    "explicit-share.png",
                    "shared-visibility.png",
                    "public-visibility.png",
                    "own-uncategorized.png",
                },
            ),
            (
                "admin",
                True,
                {
                    "own-private.png",
                    "explicit-share.png",
                    "shared-visibility.png",
                    "public-visibility.png",
                    "other-private.png",
                    "own-uncategorized.png",
                    "other-uncategorized.png",
                },
            ),
        ],
    )
    def test_all_scope_authorization_and_counts_are_consistent(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
        user_id: str,
        is_admin: bool,
        expected: set[str],
    ) -> None:
        self._seed_visibility_matrix(stores)
        image_store = stores[0]
        with image_store._db.transaction() as cursor:
            cursor.executemany(
                "UPDATE images SET starred = TRUE WHERE image_name = ?",
                [("own-private.png",), ("other-private.png",)],
            )

        dtos = image_store.get_many(limit=100, board_id="all", user_id=user_id, is_admin=is_admin)
        names = image_store.get_image_names(board_id="all", user_id=user_id, is_admin=is_admin)

        assert {image.image_name for image in dtos.items} == expected
        assert set(names.image_names) == expected
        assert dtos.total == names.total_count == len(expected)
        assert names.starred_count == len(expected & {"own-private.png", "other-private.png"})

    def test_all_scope_combines_with_inclusive_date_filters(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        self._seed_visibility_matrix(stores)
        image_store = stores[0]
        _set_created_at(image_store, "own-private.png", "2026-06-01 00:00:00.000")
        _set_created_at(image_store, "explicit-share.png", "2026-06-01 23:59:59.999")
        _set_created_at(image_store, "public-visibility.png", "2026-06-02 00:00:00.000")

        dtos = image_store.get_many(
            limit=100,
            board_id="all",
            user_id="user1",
            created_from="2026-06-01",
            created_to="2026-06-01",
        )
        names = image_store.get_image_names(
            board_id="all",
            user_id="user1",
            created_from="2026-06-01",
            created_to="2026-06-01",
        )

        assert {image.image_name for image in dtos.items} == {"own-private.png", "explicit-share.png"}
        assert set(names.image_names) == {"own-private.png", "explicit-share.png"}
        assert dtos.total == names.total_count == 2

    @pytest.mark.parametrize(("user_id", "is_admin"), [("user1", False), ("admin", True)])
    def test_all_scope_excludes_dangling_board_associations(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
        user_id: str,
        is_admin: bool,
    ) -> None:
        image_store = stores[0]
        _save(image_store, "dangling.png", user_id="user1")
        # Production foreign keys prevent this state, but imported/legacy DBs
        # may contain it. Seed it deliberately to lock down fail-closed reads.
        image_store._db._conn.execute("PRAGMA foreign_keys = OFF")
        image_store._db._conn.execute(
            "INSERT INTO board_images (board_id, image_name) VALUES (?, ?)",
            ("deleted-board", "dangling.png"),
        )
        image_store._db._conn.commit()
        image_store._db._conn.execute("PRAGMA foreign_keys = ON")

        dtos = image_store.get_many(limit=100, board_id="all", user_id=user_id, is_admin=is_admin)
        names = image_store.get_image_names(board_id="all", user_id=user_id, is_admin=is_admin)

        assert dtos.items == []
        assert dtos.total == 0
        assert names.image_names == []
        assert names.total_count == names.starred_count == 0


def _set_created_at(store: SqliteImageRecordStorage, name: str, created_at: str) -> None:
    """created_at is written by a SQL column default; tests override it directly."""
    with store._db.transaction() as cursor:
        cursor.execute("UPDATE images SET created_at = ? WHERE image_name = ?", (created_at, name))


class TestCreatedAtRangeFiltering:
    """get_many()/get_image_names() filter by inclusive created_from/created_to dates.

    Bounds are date-only strings compared lexicographically against the ISO text
    column, which stores both space- and T-separated timestamps.
    """

    def _seed_dated(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "jan30.png")
        _save(store, "jan31-morning.png")
        _save(store, "jan31-last-second.png")
        _save(store, "feb01-midnight.png")
        _save(store, "feb15-t-sep.png")
        _set_created_at(store, "jan30.png", "2026-01-30 12:00:00.000")
        _set_created_at(store, "jan31-morning.png", "2026-01-31 08:30:00.000")
        _set_created_at(store, "jan31-last-second.png", "2026-01-31 23:59:59.999")
        _set_created_at(store, "feb01-midnight.png", "2026-02-01 00:00:00.000")
        _set_created_at(store, "feb15-t-sep.png", "2026-02-15T10:00:00.000")

    def test_created_from_is_inclusive(self, store: SqliteImageRecordStorage) -> None:
        self._seed_dated(store)

        result = store.get_many(limit=10, created_from="2026-01-31")

        assert {r.image_name for r in result.items} == {
            "jan31-morning.png",
            "jan31-last-second.png",
            "feb01-midnight.png",
            "feb15-t-sep.png",
        }

    def test_created_to_includes_end_of_day_and_excludes_next_midnight(self, store: SqliteImageRecordStorage) -> None:
        self._seed_dated(store)

        result = store.get_many(limit=10, created_to="2026-01-31")

        assert {r.image_name for r in result.items} == {
            "jan30.png",
            "jan31-morning.png",
            "jan31-last-second.png",
        }

    def test_created_to_handles_month_rollover(self, store: SqliteImageRecordStorage) -> None:
        """created_to on the last day of a month must not lexicographically leak into the next month."""
        self._seed_dated(store)

        result = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-01-31")

        assert {r.image_name for r in result.items} == {"jan31-morning.png", "jan31-last-second.png"}

    def test_range_matches_t_separated_timestamps(self, store: SqliteImageRecordStorage) -> None:
        self._seed_dated(store)

        result = store.get_many(limit=10, created_from="2026-02-15", created_to="2026-02-15")

        assert {r.image_name for r in result.items} == {"feb15-t-sep.png"}

    def test_range_combines_with_search_term(self, store: SqliteImageRecordStorage) -> None:
        self._seed_dated(store)

        result = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-02-01", search_term="feb01")

        assert {r.image_name for r in result.items} == set()
        # search_term matches metadata/created_at, not names; a created_at match works
        result = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-02-01", search_term="2026-02-01")
        assert {r.image_name for r in result.items} == {"feb01-midnight.png"}

    def test_range_combines_with_board_filter(
        self,
        stores: tuple[SqliteImageRecordStorage, SqliteBoardRecordStorage, SqliteBoardImageRecordStorage],
    ) -> None:
        image_store, board_store, board_image_store = stores
        self._seed_dated(image_store)
        board = board_store.save(board_name="Dated Board", user_id="user1")
        board_image_store.add_image_to_board(board_id=board.board_id, image_name="jan30.png")
        board_image_store.add_image_to_board(board_id=board.board_id, image_name="feb01-midnight.png")

        result = image_store.get_many(limit=10, board_id=board.board_id, created_from="2026-02-01")

        assert {r.image_name for r in result.items} == {"feb01-midnight.png"}

    def test_get_many_total_and_get_image_names_counts_are_consistent(self, store: SqliteImageRecordStorage) -> None:
        self._seed_dated(store)
        with store._db.transaction() as cursor:
            cursor.execute("UPDATE images SET starred = TRUE WHERE image_name = ?", ("jan31-morning.png",))

        dtos = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-02-01")
        names = store.get_image_names(created_from="2026-01-31", created_to="2026-02-01")

        assert dtos.total == names.total_count == 3
        assert set(names.image_names) == {r.image_name for r in dtos.items}
        assert names.starred_count == 1

    def test_no_range_returns_everything(self, store: SqliteImageRecordStorage) -> None:
        self._seed_dated(store)

        result = store.get_many(limit=10)

        assert result.total == 5


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
