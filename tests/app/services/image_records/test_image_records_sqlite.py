"""DB-backed tests for SqliteImageRecordStorage.

Verifies that image_subfolder round-trips correctly through save(), get(),
get_many(), and delete_intermediates() against a real (in-memory) SQLite database,
and that get_many()/get_image_names() enforce per-user ownership isolation.
"""

import pytest

from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
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
) -> None:
    store.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=is_intermediate,
        image_subfolder=subfolder,
        user_id=user_id,
    )


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


class TestDeleteIntermediatesSubfolder:
    """delete_intermediates() returns (name, subfolder) pairs and removes rows."""

    def test_returns_subfolder_pairs(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "keep.png", subfolder="general", is_intermediate=False)
        _save(store, "tmp1.png", subfolder="intermediate", is_intermediate=True)
        _save(store, "tmp2.png", subfolder="intermediate", is_intermediate=True)

        pairs = store.delete_intermediates()

        # Should return only intermediate images with their subfolders
        assert len(pairs) == 2
        names_and_subs = set(pairs)
        assert ("tmp1.png", "intermediate") in names_and_subs
        assert ("tmp2.png", "intermediate") in names_and_subs

        # Non-intermediate image should still exist
        record = store.get("keep.png")
        assert record.image_subfolder == "general"

    def test_intermediates_are_deleted(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        store.delete_intermediates()

        from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException

        with pytest.raises(ImageRecordNotFoundException):
            store.get("tmp.png")


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
