"""Image records on every database backend.

Verifies that image_subfolder round-trips through save(), get(), get_many() and get_subfolders(), that
get_many()/get_image_names() enforce per-user ownership isolation, and how intermediates are deleted.
"""

from collections.abc import Sequence
from typing import Any, Optional

import pytest
from sqlalchemy import insert, select, true, update
from sqlalchemy.exc import DBAPIError

from invokeai.app.invocations.fields import MetadataFieldValidator
from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecordChanges,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries import images as image_queries
from invokeai.app.services.shared.database.schema.boards import shared_boards
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.pagination import SQLiteDirection
from tests.fixtures.database import capture_statements, explain_query_plan
from tests.fixtures.races import while_in_flight
from tests.fixtures.sqlite_database import sqlite_cursor

Stores = tuple[ImageRecordStorage, BoardRecordStorage, BoardImageRecordStorage]


@pytest.fixture
def store(database: Database) -> ImageRecordStorage:
    return ImageRecordStorage(database)


@pytest.fixture
def stores(database: Database) -> Stores:
    """Image, board, and board-image storages sharing one database."""
    return ImageRecordStorage(database), BoardRecordStorage(database), BoardImageRecordStorage(database)


def _save(
    store: ImageRecordStorage,
    name: str,
    subfolder: str = "",
    is_intermediate: bool = False,
    user_id: Optional[str] = None,
    category: ImageCategory = ImageCategory.GENERAL,
    metadata: Optional[str] = None,
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
        metadata=metadata,
    )


def _set_created_at(database: Database, name: str, created_at: str) -> None:
    """created_at is written when the record is saved; tests set it directly."""
    with database.begin(write=True) as conn:
        conn.execute(update(images).where(images.c.image_name == name).values(created_at=created_at))


def _intermediates(database: Database) -> list[tuple[str, str]]:
    with database.begin(write=False) as conn:
        statement = select(images.c.image_name, images.c.image_subfolder).where(images.c.is_intermediate == true())
        return [(row[0], row[1]) for row in conn.execute(statement.order_by(images.c.image_name))]


def _capture_names_plan(
    database: Database, store: ImageRecordStorage, **kwargs: Any
) -> tuple[ImageNamesResult, str, list[str]]:
    with capture_statements(database) as statements:
        result = store.get_image_names(**kwargs)
    statement, parameters = next(
        (statement, parameters)
        for statement, parameters in statements
        if statement.lstrip().startswith("SELECT images.image_name")
    )
    return result, statement, explain_query_plan(database, statement, parameters)


class TestImageSubfolderRoundTrip:
    """save() -> get() preserves image_subfolder."""

    def test_default_empty_subfolder(self, store: ImageRecordStorage) -> None:
        _save(store, "img_default.png")
        record = store.get("img_default.png")
        assert record.image_subfolder == ""

    def test_custom_subfolder(self, store: ImageRecordStorage) -> None:
        _save(store, "img_sub.png", subfolder="2026/04/11")
        record = store.get("img_sub.png")
        assert record.image_subfolder == "2026/04/11"

    def test_nested_subfolder(self, store: ImageRecordStorage) -> None:
        _save(store, "img_nested.png", subfolder="a/b/c/d")
        record = store.get("img_nested.png")
        assert record.image_subfolder == "a/b/c/d"


class TestGetManySubfolder:
    """get_many() deserializes image_subfolder for every row."""

    def test_get_many_returns_subfolders(self, store: ImageRecordStorage) -> None:
        _save(store, "flat.png", subfolder="")
        _save(store, "dated.png", subfolder="2026/01")
        _save(store, "hashed.png", subfolder="ab")

        result = store.get_many(limit=10, order_dir=SQLiteDirection.Ascending)
        by_name = {r.image_name: r.image_subfolder for r in result.items}

        assert by_name["flat.png"] == ""
        assert by_name["dated.png"] == "2026/01"
        assert by_name["hashed.png"] == "ab"


class TestImageRecordExists:
    def test_exists_returns_true_for_saved_image(self, store: ImageRecordStorage) -> None:
        _save(store, "exists.png")

        assert store.exists("exists.png") is True

    def test_exists_returns_false_for_missing_image(self, store: ImageRecordStorage) -> None:
        assert store.exists("missing.png") is False


class TestSave:
    def test_saving_a_taken_name_keeps_the_record_and_returns_its_time(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        _save(store, "taken.png", subfolder="first", user_id="alice")
        _set_created_at(database, "taken.png", "2026-01-02 03:04:05.678")

        created_at = store.save(
            image_name="taken.png",
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=ImageCategory.USER,
            width=8,
            height=8,
            has_workflow=True,
            image_subfolder="second",
            user_id="bob",
        )

        assert created_at.isoformat(sep=" ", timespec="milliseconds") == "2026-01-02 03:04:05.678"
        record = store.get("taken.png")
        assert (record.image_subfolder, record.image_origin, store.get_user_id("taken.png")) == (
            "first",
            ResourceOrigin.INTERNAL,
            "alice",
        )

    def test_an_image_without_an_owner_belongs_to_the_system(self, store: ImageRecordStorage) -> None:
        _save(store, "unowned.png")

        assert store.get_user_id("unowned.png") == "system"
        assert store.get_user_id("missing.png") is None


class TestFields:
    def test_save_stores_every_field_and_update_changes_every_field(self, store: ImageRecordStorage) -> None:
        store.save(
            image_name="full.png",
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=ImageCategory.CONTROL,
            width=640,
            height=480,
            has_workflow=True,
            is_intermediate=True,
            starred=True,
            session_id="session-1",
            node_id="node-1",
            metadata='{"seed": 1}',
            user_id="user-1",
            image_subfolder="sub/dir",
            project_id="project-1",
        )

        saved = store.get("full.png")
        store.update(
            "full.png",
            ImageRecordChanges(
                image_category=ImageCategory.USER, session_id="session-2", is_intermediate=False, starred=False
            ),
        )
        updated = store.get("full.png")

        assert (
            saved.image_origin,
            saved.image_category,
            saved.width,
            saved.height,
            saved.has_workflow,
            saved.is_intermediate,
            saved.starred,
            saved.session_id,
            saved.node_id,
            saved.image_subfolder,
            saved.project_id,
        ) == (
            ResourceOrigin.EXTERNAL,
            ImageCategory.CONTROL,
            640,
            480,
            True,
            True,
            True,
            "session-1",
            "node-1",
            "sub/dir",
            "project-1",
        )
        assert store.get_user_id("full.png") == "user-1"
        assert store.get_metadata("full.png") == MetadataFieldValidator.validate_json('{"seed": 1}')
        assert (updated.image_category, updated.session_id, updated.is_intermediate, updated.starred) == (
            ImageCategory.USER,
            "session-2",
            False,
            False,
        )
        assert updated.node_id == "node-1" and updated.has_workflow is True


class TestFilters:
    def test_the_origin_filter_narrows_pages_and_names_alike(self, store: ImageRecordStorage) -> None:
        _save(store, "internal.png")
        store.save(
            image_name="external.png",
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=ImageCategory.GENERAL,
            width=8,
            height=8,
            has_workflow=False,
        )

        page = store.get_many(limit=10, image_origin=ResourceOrigin.EXTERNAL)
        names = store.get_image_names(image_origin=ResourceOrigin.EXTERNAL)

        assert [r.image_name for r in page.items] == names.image_names == ["external.png"]
        assert page.total == 1

    def test_a_size_backfill_fills_only_sizes_not_known_yet(self, store: ImageRecordStorage) -> None:
        _save(store, "unknown.png")
        _save(store, "measured.png")
        store.set_file_size_bytes("measured.png", 100)

        store.set_file_sizes_bytes({"unknown.png": 7, "measured.png": 9})

        assert [store.get(name).file_size_bytes for name in ("unknown.png", "measured.png")] == [7, 100]


class TestGetSubfolders:
    """get_subfolders() maps the named rows to their on-disk subfolders without touching them."""

    def test_returns_subfolders_of_existing_rows_only(self, store: ImageRecordStorage) -> None:
        _save(store, "keep.png", subfolder="general", is_intermediate=False)
        _save(store, "tmp1.png", subfolder="intermediate", is_intermediate=True)

        assert store.get_subfolders(["keep.png", "tmp1.png", "missing.png"]) == {
            "keep.png": "general",
            "tmp1.png": "intermediate",
        }
        assert store.get("tmp1.png").image_subfolder == "intermediate"

    def test_intermediates_are_deleted_via_delete_intermediates_by_names(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        deleted = store.delete_intermediates_by_names([name for name, _ in _intermediates(database)])

        assert deleted == ["tmp.png"]
        with pytest.raises(ImageRecordNotFoundException):
            store.get("tmp.png")


@pytest.mark.sqlite_only  # A broken table would stay broken for every later test on a shared server schema.
class TestQueryFaultsAreNotNotFound:
    """A failing query means the database is unavailable, not that the image is missing.

    Reporting a query fault as "not found" propagates all the way to the API, where it becomes a 404
    and tells the frontend to drop a live image from its cache.
    """

    def _break_the_images_table(self, database: Database) -> None:
        with database.begin(write=True) as conn:
            conn.exec_driver_sql("ALTER TABLE images RENAME TO images_moved")

    def test_get_raises_the_db_error_not_not_found(self, database: Database, store: ImageRecordStorage) -> None:
        _save(store, "live.png")
        self._break_the_images_table(database)

        with pytest.raises(DBAPIError):
            store.get("live.png")

    def test_get_metadata_raises_the_db_error_not_not_found(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        _save(store, "live.png")
        self._break_the_images_table(database)

        with pytest.raises(DBAPIError):
            store.get_metadata("live.png")

    def test_missing_row_still_raises_not_found(self, store: ImageRecordStorage) -> None:
        """The genuine not-found path is untouched."""
        with pytest.raises(ImageRecordNotFoundException):
            store.get("never-existed.png")
        with pytest.raises(ImageRecordNotFoundException):
            store.get_metadata("never-existed.png")


class TestDeleteIntermediatesByNames:
    """delete_intermediates_by_names() deletes only rows that are still intermediates."""

    def test_promoted_image_keeps_its_record(self, database: Database, store: ImageRecordStorage) -> None:
        """An image promoted out of intermediate status after the snapshot must survive."""
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        _save(store, "promoted.png", subfolder="x", is_intermediate=True)
        snapshot = [name for name, _ in _intermediates(database)]
        assert set(snapshot) == {"tmp.png", "promoted.png"}

        # Simulate the race: the image stops being an intermediate between the snapshot and delete.
        store.update("promoted.png", ImageRecordChanges(is_intermediate=False))

        deleted = store.delete_intermediates_by_names(snapshot)

        assert deleted == ["tmp.png"]
        # promoted.png is excluded from the returned names, so the caller never purges its files.
        assert store.get("promoted.png").is_intermediate is False
        with pytest.raises(ImageRecordNotFoundException):
            store.get("tmp.png")

    def test_a_promotion_in_flight_is_waited_for_and_keeps_the_record(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        """A delete that meets a promotion still in flight waits for it, and then keeps the promoted record."""
        _save(store, "tmp.png", is_intermediate=True)
        _save(store, "promoted.png", is_intermediate=True)
        deleted: list[str] = []

        def promote(q: Queries) -> None:
            q.images.update("promoted.png", ImageRecordChanges(is_intermediate=False))

        errors = while_in_flight(
            database, promote, lambda: deleted.extend(store.delete_intermediates_by_names(["tmp.png", "promoted.png"]))
        )

        assert errors == []
        assert deleted == ["tmp.png"]
        assert store.get("promoted.png").is_intermediate is False

    def test_unknown_and_empty_names_are_no_ops(self, store: ImageRecordStorage) -> None:
        _save(store, "keep.png", is_intermediate=False)

        assert store.delete_intermediates_by_names([]) == []
        # "gone.png" has no record at all and "keep.png" is not an intermediate, so neither is
        # deleted or returned; keep.png must still be present afterwards.
        assert store.delete_intermediates_by_names(["gone.png", "keep.png"]) == []
        assert store.get("keep.png").image_name == "keep.png"

    def test_names_spanning_several_statements(
        self, database: Database, store: ImageRecordStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunking must not lose rows: a name list spanning several chunks."""
        monkeypatch.setattr(image_queries, "IN_CHUNK", 3)
        names = [f"tmp{i:02d}.png" for i in range(3 * 2 + 1)]
        for name in names:
            _save(store, name, is_intermediate=True)
        # One image in the middle chunk is promoted and must survive.
        survivor = names[4]
        store.update(survivor, ImageRecordChanges(is_intermediate=False))

        deleted = store.delete_intermediates_by_names(names[::-1])

        assert deleted == [name for name in names[::-1] if name != survivor]
        assert store.get(survivor).is_intermediate is False
        assert _intermediates(database) == []

    @pytest.mark.parametrize("operation", ["delete_intermediates_by_names", "delete_many", "get_subfolders"])
    def test_every_name_is_covered_and_no_statement_binds_more_than_sqlite_takes(
        self, database: Database, store: ImageRecordStorage, operation: str
    ) -> None:
        """999 is the SQLITE_MAX_VARIABLE_NUMBER default on builds older than 3.32."""
        names = [f"tmp{i:05d}.png" for i in range(2 * 500 + 7)]
        with database.begin(write=True) as conn:
            conn.execute(
                insert(images),
                [
                    {
                        "image_name": name,
                        "image_origin": ResourceOrigin.INTERNAL.value,
                        "image_category": ImageCategory.GENERAL.value,
                        "width": 8,
                        "height": 8,
                        "is_intermediate": True,
                        "image_subfolder": "sub",
                    }
                    for name in names
                ],
            )

        with capture_statements(database) as statements:
            result = getattr(store, operation)(names)

        widest = max(len(parameters) for _, parameters in statements)
        assert 0 < widest <= 999
        if operation == "get_subfolders":
            assert result == dict.fromkeys(names, "sub")
        else:
            assert store.get_image_names().image_names == []
        if operation == "delete_intermediates_by_names":
            assert result == names

    def test_a_guard_narrows_what_is_deleted(self, store: ImageRecordStorage) -> None:
        _save(store, "a.png", is_intermediate=True)
        _save(store, "b.png", is_intermediate=True)
        asked: list[list[str]] = []

        def guard(q: Queries, names: Sequence[str]) -> list[str]:
            asked.append(list(names))
            return [name for name in names if name != "b.png"]

        assert store.delete_intermediates_by_names(["a.png", "b.png"], guard) == ["a.png"]
        assert asked == [["a.png", "b.png"]]
        assert store.get("b.png").is_intermediate is True


class TestDeleteMany:
    def test_names_spanning_several_statements_are_all_deleted(
        self, database: Database, store: ImageRecordStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(image_queries, "IN_CHUNK", 3)
        names = [f"img{i:02d}.png" for i in range(3 * 2 + 1)]
        for name in [*names, "kept.png"]:
            _save(store, name)

        store.delete_many(names)

        assert store.get_image_names(starred_first=False).image_names == ["kept.png"]


class TestBoardCover:
    def test_the_cover_is_the_newest_image_that_is_no_intermediate_a_starred_one_first(
        self, database: Database, stores: Stores
    ) -> None:
        image_store, board_store, board_image_store = stores
        board = board_store.save(board_name="Board", user_id="system")
        for name, created_at, starred, intermediate in (
            ("old-starred.png", "2026-01-01 00:00:00.000", True, False),
            ("newer.png", "2026-01-02 00:00:00.000", False, False),
            ("newest-starred-intermediate.png", "2026-01-03 00:00:00.000", True, True),
        ):
            _save(image_store, name, is_intermediate=intermediate)
            image_store.update(name, ImageRecordChanges(starred=starred))
            _set_created_at(database, name, created_at)
            board_image_store.add_image_to_board(board_id=board.board_id, image_name=name)

        cover = image_store.get_most_recent_image_for_board(board.board_id)

        assert cover is not None and cover.image_name == "old-starred.png"
        assert image_store.get_most_recent_image_for_board(board_store.save("Empty", "system").board_id) is None


class TestSearch:
    def test_search_ignores_case_and_matches_percent_and_underscore_literally(self, store: ImageRecordStorage) -> None:
        _save(store, "literal.png", metadata='{"positive_prompt": "100% Cotton_shirt"}')
        # Matched too if `%` and `_` in the term were wildcards.
        _save(store, "wildcard.png", metadata='{"positive_prompt": "100 cotton shirts"}')

        result = store.get_image_names(search_term="100% COTTON_")

        assert result.image_names == ["literal.png"]
        assert store.get_many(limit=10, search_term="100% COTTON_").total == 1


class TestSearchBeyondAscii:
    def test_a_search_ignores_the_case_of_letters_beyond_ascii(self, store: ImageRecordStorage) -> None:
        _save(store, "apples.png", metadata='{"prompt": "frische äpfel"}')

        assert store.get_image_names(search_term="ÄPFEL").image_names == ["apples.png"]


class TestOwnershipFilteringOmittedBoard:
    """get_many()/get_image_names() enforce per-user isolation when board_id is omitted.

    Without this, a non-admin could enumerate every user's images (including images
    on other users' private boards) simply by omitting the board_id query parameter.
    """

    def _seed_two_users(self, stores: Stores) -> str:
        """user1: one image on a private board + one uncategorized. user2: one uncategorized."""
        image_store, board_store, board_image_store = stores
        _save(image_store, "u1-boarded.png", user_id="user1")
        _save(image_store, "u1-uncat.png", user_id="user1")
        _save(image_store, "u2-uncat.png", user_id="user2")
        board = board_store.save(board_name="User1 Private Board", user_id="user1")
        board_image_store.add_image_to_board(board_id=board.board_id, image_name="u1-boarded.png")
        return board.board_id

    def test_get_many_omitted_board_filters_by_owner(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id="user2", is_admin=False)

        assert {r.image_name for r in result.items} == {"u2-uncat.png"}
        assert result.total == 1

    def test_get_many_omitted_board_owner_sees_boarded_and_uncategorized(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id="user1", is_admin=False)

        assert {r.image_name for r in result.items} == {"u1-boarded.png", "u1-uncat.png"}
        assert result.total == 2

    def test_get_many_omitted_board_admin_sees_all(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id="admin", is_admin=True)

        assert {r.image_name for r in result.items} == {"u1-boarded.png", "u1-uncat.png", "u2-uncat.png"}
        assert result.total == 3

    def test_get_many_omitted_board_single_user_mode_sees_all(self, stores: Stores) -> None:
        """user_id=None (single-user mode) applies no ownership filter."""
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, user_id=None, is_admin=False)

        assert result.total == 3

    @pytest.mark.parametrize(("user_id", "is_admin"), [("admin", True), (None, False)])
    def test_get_many_none_board_unscoped_lists_every_owners_unboarded_images(
        self, stores: Stores, user_id: Optional[str], is_admin: bool
    ) -> None:
        self._seed_two_users(stores)

        result = stores[0].get_many(limit=10, board_id="none", user_id=user_id, is_admin=is_admin)

        assert {r.image_name for r in result.items} == {"u1-uncat.png", "u2-uncat.png"}

    def test_get_many_none_board_still_filters_by_owner(self, stores: Stores) -> None:
        """board_id="none" (uncategorized) keeps its existing per-user isolation."""
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, board_id="none", user_id="user1", is_admin=False)

        assert {r.image_name for r in result.items} == {"u1-uncat.png"}

    def test_get_many_explicit_board_returns_board_contents(self, stores: Stores) -> None:
        """An explicit board_id lists that board's images; read access is the router's job."""
        board_id = self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_many(limit=10, board_id=board_id, user_id="user1", is_admin=False)

        assert {r.image_name for r in result.items} == {"u1-boarded.png"}

    def test_get_image_names_omitted_board_filters_by_owner(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(user_id="user2", is_admin=False)

        assert result.image_names == ["u2-uncat.png"]
        assert result.total_count == 1

    def test_get_image_names_omitted_board_admin_sees_all(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(user_id="admin", is_admin=True)

        assert set(result.image_names) == {"u1-boarded.png", "u1-uncat.png", "u2-uncat.png"}
        assert result.total_count == 3

    def test_get_image_names_omitted_board_single_user_mode_sees_all(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(user_id=None, is_admin=False)

        assert result.total_count == 3

    def test_get_image_names_none_board_still_filters_by_owner(self, stores: Stores) -> None:
        self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(board_id="none", user_id="user1", is_admin=False)

        assert result.image_names == ["u1-uncat.png"]

    def test_get_image_names_explicit_board_returns_board_contents(self, stores: Stores) -> None:
        board_id = self._seed_two_users(stores)
        image_store = stores[0]

        result = image_store.get_image_names(board_id=board_id, user_id="user1", is_admin=False)

        assert result.image_names == ["u1-boarded.png"]


class TestAllReadableBoardsFiltering:
    def _seed_visibility_matrix(self, database: Database, stores: Stores) -> None:
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

        with database.begin(write=True) as conn:
            conn.execute(insert(users).values(user_id="user1", email="user1@example.com", password_hash="unused"))
            conn.execute(insert(shared_boards).values(board_id=shared_private_id, user_id="user1"))

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
            ("user3", False, {"shared-visibility.png", "public-visibility.png"}),
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
        database: Database,
        stores: Stores,
        user_id: str,
        is_admin: bool,
        expected: set[str],
    ) -> None:
        self._seed_visibility_matrix(database, stores)
        image_store = stores[0]
        for name in ("own-private.png", "other-private.png"):
            image_store.update(name, ImageRecordChanges(starred=True))

        dtos = image_store.get_many(limit=100, board_id="all", user_id=user_id, is_admin=is_admin)
        names = image_store.get_image_names(board_id="all", user_id=user_id, is_admin=is_admin)

        assert {image.image_name for image in dtos.items} == expected
        assert set(names.image_names) == expected
        assert dtos.total == names.total_count == len(expected)
        assert names.starred_count == len(expected & {"own-private.png", "other-private.png"})

    def test_all_scope_combines_with_inclusive_date_filters(self, database: Database, stores: Stores) -> None:
        self._seed_visibility_matrix(database, stores)
        image_store = stores[0]
        _set_created_at(database, "own-private.png", "2026-06-01 00:00:00.000")
        _set_created_at(database, "explicit-share.png", "2026-06-01 23:59:59.999")
        _set_created_at(database, "public-visibility.png", "2026-06-02 00:00:00.000")

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

    @pytest.mark.sqlite_only  # Foreign keys are switched off for it, as only a migrated SQLite database can need.
    @pytest.mark.parametrize(("user_id", "is_admin"), [("user1", False), ("admin", True)])
    def test_all_scope_excludes_dangling_board_associations(
        self,
        database: Database,
        stores: Stores,
        user_id: str,
        is_admin: bool,
    ) -> None:
        image_store = stores[0]
        _save(image_store, "dangling.png", user_id="user1")
        # Production foreign keys prevent this state, but imported/legacy DBs
        # may contain it. Seed it deliberately to lock down fail-closed reads.
        # (A PRAGMA takes effect only outside a transaction, so it goes to the connection itself.)
        database.sqlite.conn.execute("PRAGMA foreign_keys = OFF")
        with sqlite_cursor(database) as cursor:
            cursor.execute(
                "INSERT INTO board_images (board_id, image_name) VALUES (?, ?)", ("deleted-board", "dangling.png")
            )
        database.sqlite.conn.execute("PRAGMA foreign_keys = ON")

        dtos = image_store.get_many(limit=100, board_id="all", user_id=user_id, is_admin=is_admin)
        names = image_store.get_image_names(board_id="all", user_id=user_id, is_admin=is_admin)

        assert dtos.items == []
        assert dtos.total == 0
        assert names.image_names == []
        assert names.total_count == names.starred_count == 0


class TestCreatedAtRangeFiltering:
    """get_many()/get_image_names() filter by inclusive created_from/created_to dates.

    Bounds are date-only strings compared lexicographically against the ISO text
    column, which stores both space- and T-separated timestamps.
    """

    def _seed_dated(self, database: Database, store: ImageRecordStorage) -> None:
        dated = {
            "jan30.png": "2026-01-30 12:00:00.000",
            "jan31-morning.png": "2026-01-31 08:30:00.000",
            "jan31-last-second.png": "2026-01-31 23:59:59.999",
            "feb01-midnight.png": "2026-02-01 00:00:00.000",
            "feb15-t-sep.png": "2026-02-15T10:00:00.000",
        }
        for name, created_at in dated.items():
            _save(store, name)
            _set_created_at(database, name, created_at)

    def test_created_from_is_inclusive(self, database: Database, store: ImageRecordStorage) -> None:
        self._seed_dated(database, store)

        result = store.get_many(limit=10, created_from="2026-01-31")

        assert {r.image_name for r in result.items} == {
            "jan31-morning.png",
            "jan31-last-second.png",
            "feb01-midnight.png",
            "feb15-t-sep.png",
        }

    def test_created_to_includes_end_of_day_and_excludes_next_midnight(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        self._seed_dated(database, store)

        result = store.get_many(limit=10, created_to="2026-01-31")

        assert {r.image_name for r in result.items} == {
            "jan30.png",
            "jan31-morning.png",
            "jan31-last-second.png",
        }

    def test_created_to_handles_month_rollover(self, database: Database, store: ImageRecordStorage) -> None:
        """created_to on the last day of a month must not lexicographically leak into the next month."""
        self._seed_dated(database, store)

        result = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-01-31")

        assert {r.image_name for r in result.items} == {"jan31-morning.png", "jan31-last-second.png"}

    def test_range_matches_t_separated_timestamps(self, database: Database, store: ImageRecordStorage) -> None:
        self._seed_dated(database, store)

        result = store.get_many(limit=10, created_from="2026-02-15", created_to="2026-02-15")

        assert {r.image_name for r in result.items} == {"feb15-t-sep.png"}

    def test_range_combines_with_search_term(self, database: Database, store: ImageRecordStorage) -> None:
        self._seed_dated(database, store)

        result = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-02-01", search_term="feb01")

        assert {r.image_name for r in result.items} == set()
        # search_term matches metadata/created_at, not names; a created_at match works
        result = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-02-01", search_term="2026-02-01")
        assert {r.image_name for r in result.items} == {"feb01-midnight.png"}

    def test_range_combines_with_board_filter(self, database: Database, stores: Stores) -> None:
        image_store, board_store, board_image_store = stores
        self._seed_dated(database, image_store)
        board = board_store.save(board_name="Dated Board", user_id="user1")
        board_image_store.add_image_to_board(board_id=board.board_id, image_name="jan30.png")
        board_image_store.add_image_to_board(board_id=board.board_id, image_name="feb01-midnight.png")

        result = image_store.get_many(limit=10, board_id=board.board_id, created_from="2026-02-01")

        assert {r.image_name for r in result.items} == {"feb01-midnight.png"}

    def test_get_many_total_and_get_image_names_counts_are_consistent(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        self._seed_dated(database, store)
        store.update("jan31-morning.png", ImageRecordChanges(starred=True))

        dtos = store.get_many(limit=10, created_from="2026-01-31", created_to="2026-02-01")
        names = store.get_image_names(created_from="2026-01-31", created_to="2026-02-01")

        assert dtos.total == names.total_count == 3
        assert set(names.image_names) == {r.image_name for r in dtos.items}
        assert names.starred_count == 1

    def test_no_range_returns_everything(self, database: Database, store: ImageRecordStorage) -> None:
        self._seed_dated(database, store)

        result = store.get_many(limit=10)

        assert result.total == 5

    def test_names_by_date_are_the_days_images_that_are_not_intermediates(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        self._seed_dated(database, store)
        _save(store, "jan31-intermediate.png", is_intermediate=True)
        _set_created_at(database, "jan31-intermediate.png", "2026-01-31 09:00:00.000")

        result = store.get_image_names_by_date("2026-01-31", starred_first=False, order_dir=SQLiteDirection.Ascending)

        assert result.image_names == ["jan31-morning.png", "jan31-last-second.png"]
        # A date that is no ISO day matches nothing, as SQLite's DATE() did.
        assert store.get_image_names_by_date("2026-1-31").image_names == []
        assert store.get_image_names_by_date("20260131").image_names == []
        assert store.get_image_names_by_date("2026-01-31 00:00").image_names == []
        assert store.get_image_names_by_date("").image_names == []

    def test_the_last_day_there_is_matches_nothing_as_sqlite_did(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        # Its next day cannot be represented; SQLite's DATE(day, '+1 day') gave NULL.
        self._seed_dated(database, store)

        assert store.get_many(limit=10, created_to="9999-12-31").items == []
        assert store.get_image_names(created_to="9999-12-31").image_names == []


class TestOrdering:
    def test_images_of_the_same_moment_are_listed_by_name(self, database: Database, store: ImageRecordStorage) -> None:
        # Saved against name order, so that only the tie-break lists them by name.
        for name in ("c.png", "a.png", "b.png"):
            _save(store, name)
            _set_created_at(database, name, "2026-01-01 00:00:00.000")
        store.update("b.png", ImageRecordChanges(starred=True))

        def listed(starred_first: bool, order_dir: SQLiteDirection) -> list[str]:
            names = store.get_image_names(starred_first=starred_first, order_dir=order_dir).image_names
            page = [
                r.image_name for r in store.get_many(limit=10, starred_first=starred_first, order_dir=order_dir).items
            ]
            assert page == names
            return names

        assert listed(False, SQLiteDirection.Ascending) == ["a.png", "b.png", "c.png"]
        assert listed(False, SQLiteDirection.Descending) == ["c.png", "b.png", "a.png"]
        assert listed(True, SQLiteDirection.Descending) == ["b.png", "c.png", "a.png"]
        assert listed(True, SQLiteDirection.Ascending) == ["b.png", "a.png", "c.png"]

    def test_pages_walk_the_listing_and_a_page_of_none_still_counts(
        self, database: Database, store: ImageRecordStorage
    ) -> None:
        for i, name in enumerate(("c.png", "a.png", "b.png")):
            _save(store, name)
            _set_created_at(database, name, f"2026-01-0{i + 1} 00:00:00.000")
        names = store.get_image_names().image_names

        pages = [store.get_many(offset=offset, limit=1) for offset in range(3)]
        count_only = store.get_many(limit=0)

        assert [[r.image_name for r in page.items] for page in pages] == [[name] for name in names]
        assert {page.total for page in pages} == {3}
        assert count_only.items == [] and count_only.total == 3


@pytest.mark.sqlite_only
class TestGetImageNamesQueryPlans:
    def test_default_category_query_has_no_forced_index(self, database: Database, store: ImageRecordStorage) -> None:
        _save(store, "image.png", user_id="alice")

        result, statement, _ = _capture_names_plan(
            database,
            store,
            categories=[ImageCategory.GENERAL],
            is_intermediate=False,
            is_admin=True,
        )

        assert result.image_names == ["image.png"]
        assert "OUTER JOIN" not in statement
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
        self, database: Database, store: ImageRecordStorage, kwargs: dict[str, Any]
    ) -> None:
        _save(store, "image.png", user_id="alice")

        _, statement, _ = _capture_names_plan(
            database,
            store,
            categories=[ImageCategory.GENERAL],
            is_intermediate=False,
            **kwargs,
        )

        assert "OUTER JOIN" not in statement
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    def test_none_board_uses_anti_membership_filter(self, database: Database, stores: Stores) -> None:
        image_store, board_store, board_image_store = stores
        _save(image_store, "boarded.png", user_id="alice")
        _save(image_store, "unboarded.png", user_id="alice")
        board = board_store.save("Board", "alice")
        board_image_store.add_image_to_board(board.board_id, "boarded.png")

        result, statement, _ = _capture_names_plan(
            database,
            image_store,
            board_id="none",
            user_id="alice",
            is_admin=False,
            starred_first=False,
        )

        assert result.image_names == ["unboarded.png"]
        assert "NOT (EXISTS" in statement
        assert "OUTER JOIN" not in statement

    @pytest.mark.parametrize("board_id", ["none", "all"])
    def test_membership_checks_search_their_indexes(self, database: Database, stores: Stores, board_id: str) -> None:
        """A membership check that scanned a table would cost every listed image all memberships."""
        image_store, board_store, board_image_store = stores
        _save(image_store, "boarded.png", user_id="alice")
        _save(image_store, "unboarded.png", user_id="alice")
        board = board_store.save("Board", "alice")
        board_image_store.add_image_to_board(board.board_id, "boarded.png")

        _, _, plan = _capture_names_plan(database, image_store, board_id=board_id, user_id="alice", is_admin=False)

        assert any("board_images" in step for step in plan)
        scans = [step for step in plan if step.startswith("SCAN")]
        assert not [step for step in scans if any(table in step for table in ("board_images", "boards"))]

    def test_nonadmin_asset_query_has_no_forced_index(self, database: Database, store: ImageRecordStorage) -> None:
        _save(store, "asset.png", user_id="alice", category=ImageCategory.CONTROL)

        result, statement, _ = _capture_names_plan(
            database,
            store,
            categories=[ImageCategory.CONTROL],
            is_intermediate=False,
            user_id="alice",
            is_admin=False,
        )

        assert result.image_names == ["asset.png"]
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement
