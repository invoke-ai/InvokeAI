"""DB-backed tests for SqliteImageRecordStorage.

Verifies that image_subfolder round-trips correctly through save(), get(),
get_many(), and get_intermediates() against a real (in-memory) SQLite database,
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

    def test_intermediates_are_deleted_via_delete_many(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        pairs = store.get_intermediates()
        store.delete_many([name for name, _ in pairs])

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
