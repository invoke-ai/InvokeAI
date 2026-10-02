"""Regression tests for SqliteGalleryService multiuser isolation and date-based
virtual boards.

Covers JPPhoto's code-review findings (PR #9163):

1. The gallery /items/ and /items/names endpoints returned every user's items
   when ``board_id`` was omitted, because ``_build_half`` only applied a user
   filter for the explicit "none" sentinel. The fix added an ``elif user_id is
   not None and not is_admin`` branch; these tests pin the behaviour for both
   halves of the polymorphic union.

2. Date-based virtual boards were image-only: video-only dates did not appear
   at all, and mixed dates omitted videos from counts/contents/covers. The
   gallery service now owns ``get_dates`` and a ``created_date`` filter on
   ``list_item_names`` so virtual boards cover both kinds.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeVar

import pytest

from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_sqlite import SqliteBoardVideoRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.gallery.gallery_common import GalleryItemKind
from invokeai.app.services.gallery.gallery_default import SqliteGalleryService
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def services():
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    gallery = SqliteGalleryService(db=db)
    gallery.start(SimpleNamespace(services=SimpleNamespace(urls=LocalUrlService())))  # type: ignore[arg-type]
    return {
        "gallery": gallery,
        "images": SqliteImageRecordStorage(db=db),
        "videos": SqliteVideoRecordStorage(db=db),
        "boards": SqliteBoardRecordStorage(db=db),
        "board_images": SqliteBoardImageRecordStorage(db=db),
        "board_videos": SqliteBoardVideoRecordStorage(db=db),
    }


def _save_image(
    store: SqliteImageRecordStorage,
    name: str,
    user_id: str,
    category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    store.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


def _save_video(
    store: SqliteVideoRecordStorage,
    name: str,
    user_id: str,
    category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    store.save(
        video_name=name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=category,
        width=64,
        height=64,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


T = TypeVar("T")


def _capture_plan(services, call: Callable[[], T], statement_marker: str) -> tuple[T, str, list[str]]:
    db = services["images"]._db
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    try:
        result = call()
    finally:
        db._conn.set_trace_callback(None)
    statement = next(statement for statement in statements if statement_marker in statement)
    details = [row[3] for row in db._conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()]
    return result, statement, details


@pytest.fixture
def seeded(services):
    # Mixed-kind items for two users, no board association — which is the path that
    # previously bypassed user filtering entirely.
    _save_image(services["images"], "alice.png", user_id="alice")
    _save_video(services["videos"], "alice.mp4", user_id="alice")
    _save_image(services["images"], "bob.png", user_id="bob")
    _save_video(services["videos"], "bob.mp4", user_id="bob")
    return services


class TestListItemNamesOmittedBoardIdMultiuser:
    def test_non_admin_only_sees_own_items(self, seeded) -> None:
        result = seeded["gallery"].list_item_names(user_id="alice", is_admin=False)
        names = {(item.kind, item.name) for item in result.items}
        assert names == {
            (GalleryItemKind.IMAGE, "alice.png"),
            (GalleryItemKind.VIDEO, "alice.mp4"),
        }
        assert result.total_count == 2

    def test_admin_sees_all_items(self, seeded) -> None:
        result = seeded["gallery"].list_item_names(user_id="alice", is_admin=True)
        names = {(item.kind, item.name) for item in result.items}
        assert names == {
            (GalleryItemKind.IMAGE, "alice.png"),
            (GalleryItemKind.IMAGE, "bob.png"),
            (GalleryItemKind.VIDEO, "alice.mp4"),
            (GalleryItemKind.VIDEO, "bob.mp4"),
        }
        assert result.total_count == 4

    def test_explicit_none_board_still_isolates(self, seeded) -> None:
        # Before the fix this branch was correct; included here as a guard against
        # accidental regression in the still-functioning code path.
        result = seeded["gallery"].list_item_names(board_id="none", user_id="alice", is_admin=False)
        names = {(item.kind, item.name) for item in result.items}
        assert names == {
            (GalleryItemKind.IMAGE, "alice.png"),
            (GalleryItemKind.VIDEO, "alice.mp4"),
        }


def _backdate(services, table: str, name_col: str, name: str, created_at: str) -> None:
    """Rewrites created_at so tests can build multi-date galleries (save() always stamps now)."""
    db = services["images"]._db
    with db.transaction() as cursor:
        cursor.execute(f"UPDATE {table} SET created_at = ? WHERE {name_col} = ?", (created_at, name))


def _star(services, table: str, name_col: str, name: str) -> None:
    db = services["images"]._db
    with db.transaction() as cursor:
        cursor.execute(f"UPDATE {table} SET starred = 1 WHERE {name_col} = ?", (name,))


def _start_gallery_for_item_results(services) -> None:
    """Provide the URL service dependency required when gallery rows become item DTOs."""
    urls = SimpleNamespace(
        get_image_url=lambda name, thumbnail=False: f"/images/{name}{'?thumbnail=1' if thumbnail else ''}",
        get_video_url=lambda name, thumbnail=False: f"/videos/{name}{'?thumbnail=1' if thumbnail else ''}",
    )
    services["gallery"].start(SimpleNamespace(services=SimpleNamespace(urls=urls)))


def _seed_created_range(services) -> None:
    """Seed inclusive-day boundaries, excluded neighbours, and another user's matching media."""
    for name in ("before.png", "range-start.png", "range-end.png", "after.png"):
        _save_image(services["images"], name, user_id="alice")
    for name in ("before.mp4", "range-video.mp4", "range-video-end.mp4", "bob-range.mp4"):
        _save_video(services["videos"], name, user_id="bob" if name == "bob-range.mp4" else "alice")

    for table, name_col, name, created_at in [
        ("images", "image_name", "before.png", "2026-03-09 23:59:59.999"),
        ("images", "image_name", "range-start.png", "2026-03-10 00:00:00.000"),
        ("images", "image_name", "range-end.png", "2026-03-11 23:59:59.999"),
        ("images", "image_name", "after.png", "2026-03-12 00:00:00.000"),
        ("videos", "video_name", "before.mp4", "2026-03-09 23:59:59.999"),
        ("videos", "video_name", "range-video.mp4", "2026-03-10 15:00:00.000"),
        ("videos", "video_name", "range-video-end.mp4", "2026-03-11 09:00:00.000"),
        ("videos", "video_name", "bob-range.mp4", "2026-03-11 12:00:00.000"),
    ]:
        _backdate(services, table, name_col, name, created_at)


def _seed_starred(services) -> None:
    """Alice owns a starred and a plain item of each kind; Bob owns one starred image."""
    for name in ("starred.png", "plain.png"):
        _save_image(services["images"], name, user_id="alice")
    for name in ("starred.mp4", "plain.mp4"):
        _save_video(services["videos"], name, user_id="alice")
    _save_image(services["images"], "bob-starred.png", user_id="bob")

    for table, name_col, name, created_at in [
        ("images", "image_name", "starred.png", "2026-04-01 10:00:00"),
        ("videos", "video_name", "starred.mp4", "2026-04-01 11:00:00"),
        ("images", "image_name", "plain.png", "2026-04-02 10:00:00"),
        ("videos", "video_name", "plain.mp4", "2026-04-02 11:00:00"),
        ("images", "image_name", "bob-starred.png", "2026-04-02 12:00:00"),
    ]:
        _backdate(services, table, name_col, name, created_at)
    for table, name_col, name in [
        ("images", "image_name", "starred.png"),
        ("videos", "video_name", "starred.mp4"),
        ("images", "image_name", "bob-starred.png"),
    ]:
        _star(services, table, name_col, name)


class TestStarredFiltering:
    def test_list_items_starred_true_returns_only_starred_of_both_kinds_with_total(self, services) -> None:
        _seed_starred(services)
        _start_gallery_for_item_results(services)

        result = services["gallery"].list_items(limit=10, user_id="alice", is_admin=False, starred=True)

        assert [(item.kind, item.name) for item in result.items] == [
            (GalleryItemKind.VIDEO, "starred.mp4"),
            (GalleryItemKind.IMAGE, "starred.png"),
        ]
        assert result.total == 2

    def test_list_items_starred_false_returns_only_unstarred(self, services) -> None:
        _seed_starred(services)
        _start_gallery_for_item_results(services)

        result = services["gallery"].list_items(limit=10, user_id="alice", is_admin=False, starred=False)

        assert {(item.kind, item.name) for item in result.items} == {
            (GalleryItemKind.VIDEO, "plain.mp4"),
            (GalleryItemKind.IMAGE, "plain.png"),
        }
        assert result.total == 2

    def test_list_items_without_starred_is_unfiltered(self, services) -> None:
        _seed_starred(services)
        _start_gallery_for_item_results(services)

        result = services["gallery"].list_items(limit=10, user_id="alice", is_admin=False)

        assert result.total == 4

    def test_starred_filter_respects_non_admin_isolation(self, services) -> None:
        _seed_starred(services)
        gallery = services["gallery"]

        as_admin = gallery.list_item_names(user_id="alice", is_admin=True, starred=True)
        as_user = gallery.list_item_names(user_id="alice", is_admin=False, starred=True)

        assert {item.name for item in as_admin.items} == {"starred.png", "starred.mp4", "bob-starred.png"}
        assert {item.name for item in as_user.items} == {"starred.png", "starred.mp4"}

    def test_starred_filter_composes_with_board_scope(self, services) -> None:
        _seed_starred(services)
        board = services["boards"].save("Board", "alice")
        services["board_images"].add_image_to_board(board.board_id, "starred.png")
        services["board_images"].add_image_to_board(board.board_id, "plain.png")
        gallery = services["gallery"]

        on_board = gallery.list_item_names(board_id=board.board_id, user_id="alice", is_admin=False, starred=True)
        off_board = gallery.list_item_names(board_id="none", user_id="alice", is_admin=False, starred=True)

        assert [item.name for item in on_board.items] == ["starred.png"]
        assert [item.name for item in off_board.items] == ["starred.mp4"]

    def test_starred_filter_composes_with_date_filters_and_search(self, services) -> None:
        _seed_starred(services)
        gallery = services["gallery"]

        by_date = gallery.list_item_names(user_id="alice", is_admin=False, created_date="2026-04-01", starred=True)
        by_range = gallery.get_item_names(
            user_id="alice",
            is_admin=False,
            created_from="2026-04-02",
            created_to="2026-04-02",
            starred=False,
        )
        by_search = gallery.get_item_names(user_id="alice", is_admin=False, search_term="2026-04-01 11", starred=True)

        assert {item.name for item in by_date.items} == {"starred.png", "starred.mp4"}
        assert set(by_range.item_names) == {"plain.png", "plain.mp4"}
        assert by_search.item_names == ["starred.mp4"]

    def test_name_lists_keep_starred_count_semantics_under_filter(self, services) -> None:
        _seed_starred(services)
        gallery = services["gallery"]

        starred = gallery.list_item_names(user_id="alice", is_admin=False, starred=True)
        unstarred = gallery.list_item_names(user_id="alice", is_admin=False, starred=False)
        unsorted = gallery.list_item_names(user_id="alice", is_admin=False, starred=True, starred_first=False)
        flat = gallery.get_item_names(user_id="alice", is_admin=False, starred=True)

        assert starred.starred_count == starred.total_count == 2
        assert unstarred.starred_count == 0
        assert unsorted.starred_count == 0
        assert flat.item_names == [item.name for item in starred.items]

    def test_starred_first_ordering_is_unchanged_when_unfiltered(self, services) -> None:
        _seed_starred(services)

        names = [item.name for item in services["gallery"].list_item_names(user_id="alice", is_admin=False).items]

        assert names == ["starred.mp4", "starred.png", "plain.mp4", "plain.png"]


class TestGetDatesPolymorphic:
    def test_video_only_date_appears(self, services) -> None:
        # A date with videos and no images must still produce a virtual board — with the
        # video as its cover, since there is no image to fall back to.
        _save_video(services["videos"], "only.mp4", user_id="alice")
        _backdate(services, "videos", "video_name", "only.mp4", "2026-01-02 10:00:00")

        boards = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert len(boards) == 1
        board = boards[0]
        assert board.date == "2026-01-02"
        assert board.image_count == 0
        assert board.asset_count == 0
        assert board.video_count == 1
        assert board.cover_image_name is None
        assert board.cover_video_name == "only.mp4"

    def test_mixed_date_counts_both_kinds(self, services) -> None:
        _save_image(services["images"], "day1.png", user_id="alice")
        _save_video(services["videos"], "day1.mp4", user_id="alice")
        _save_video(services["videos"], "day1b.mp4", user_id="alice")
        _backdate(services, "images", "image_name", "day1.png", "2026-01-03 09:00:00")
        _backdate(services, "videos", "video_name", "day1.mp4", "2026-01-03 10:00:00")
        _backdate(services, "videos", "video_name", "day1b.mp4", "2026-01-03 11:00:00")

        boards = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert len(boards) == 1
        board = boards[0]
        assert board.date == "2026-01-03"
        assert board.image_count == 1
        assert board.video_count == 2
        # The newest item of the date is a video, so the cover is the video.
        assert board.cover_video_name == "day1b.mp4"
        assert board.cover_image_name is None

    def test_newest_image_wins_cover(self, services) -> None:
        _save_video(services["videos"], "old.mp4", user_id="alice")
        _save_image(services["images"], "new.png", user_id="alice")
        _backdate(services, "videos", "video_name", "old.mp4", "2026-01-04 09:00:00")
        _backdate(services, "images", "image_name", "new.png", "2026-01-04 10:00:00")

        boards = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert len(boards) == 1
        assert boards[0].cover_image_name == "new.png"
        assert boards[0].cover_video_name is None

    def test_dates_are_user_isolated(self, seeded) -> None:
        boards = seeded["gallery"].get_dates(user_id="alice", is_admin=False)
        # alice has one image + one video, both created today.
        assert len(boards) == 1
        assert boards[0].image_count == 1
        assert boards[0].video_count == 1

    def test_admin_sees_all_dates(self, seeded) -> None:
        boards = seeded["gallery"].get_dates(user_id="alice", is_admin=True)
        assert len(boards) == 1
        assert boards[0].image_count == 2
        assert boards[0].video_count == 2


class TestListItemNamesByCreatedDate:
    def test_returns_only_items_of_date_including_videos(self, services) -> None:
        _save_image(services["images"], "target.png", user_id="alice")
        _save_video(services["videos"], "target.mp4", user_id="alice")
        _save_image(services["images"], "other.png", user_id="alice")
        _backdate(services, "images", "image_name", "target.png", "2026-01-05 09:00:00")
        _backdate(services, "videos", "video_name", "target.mp4", "2026-01-05 10:00:00")
        _backdate(services, "images", "image_name", "other.png", "2026-01-06 09:00:00")

        result = services["gallery"].list_item_names(user_id="alice", is_admin=False, created_date="2026-01-05")

        names = {(item.kind, item.name) for item in result.items}
        assert names == {
            (GalleryItemKind.IMAGE, "target.png"),
            (GalleryItemKind.VIDEO, "target.mp4"),
        }
        assert result.total_count == 2

    def test_created_date_is_user_isolated(self, services) -> None:
        _save_video(services["videos"], "alice-day.mp4", user_id="alice")
        _save_video(services["videos"], "bob-day.mp4", user_id="bob")
        _backdate(services, "videos", "video_name", "alice-day.mp4", "2026-01-07 09:00:00")
        _backdate(services, "videos", "video_name", "bob-day.mp4", "2026-01-07 10:00:00")

        result = services["gallery"].list_item_names(user_id="alice", is_admin=False, created_date="2026-01-07")

        assert [(item.kind, item.name) for item in result.items] == [(GalleryItemKind.VIDEO, "alice-day.mp4")]


class TestCreatedRangeFiltering:
    def test_list_items_includes_utc_day_bounds_and_reports_total(self, services) -> None:
        _seed_created_range(services)
        _start_gallery_for_item_results(services)

        result = services["gallery"].list_items(
            limit=10,
            user_id="alice",
            is_admin=False,
            created_from="2026-03-10",
            created_to="2026-03-11",
        )

        assert [(item.kind, item.name) for item in result.items] == [
            (GalleryItemKind.IMAGE, "range-end.png"),
            (GalleryItemKind.VIDEO, "range-video-end.mp4"),
            (GalleryItemKind.VIDEO, "range-video.mp4"),
            (GalleryItemKind.IMAGE, "range-start.png"),
        ]
        assert result.total == 4

    def test_list_item_names_created_range_matches_item_order_and_counts(self, services) -> None:
        _seed_created_range(services)

        result = services["gallery"].list_item_names(
            user_id="alice",
            is_admin=False,
            created_from="2026-03-10",
            created_to="2026-03-11",
        )

        assert [(item.kind, item.name) for item in result.items] == [
            (GalleryItemKind.IMAGE, "range-end.png"),
            (GalleryItemKind.VIDEO, "range-video-end.mp4"),
            (GalleryItemKind.VIDEO, "range-video.mp4"),
            (GalleryItemKind.IMAGE, "range-start.png"),
        ]
        assert result.total_count == 4

    def test_get_item_names_created_range_matches_legacy_result(self, services) -> None:
        _seed_created_range(services)

        result = services["gallery"].get_item_names(
            user_id="alice",
            is_admin=False,
            created_from="2026-03-10",
            created_to="2026-03-11",
        )

        assert result.item_names == [
            "range-end.png",
            "range-video-end.mp4",
            "range-video.mp4",
            "range-start.png",
        ]
        assert result.total_count == 4

    def test_created_date_remains_an_exact_day_filter_without_ranges(self, services) -> None:
        _seed_created_range(services)

        result = services["gallery"].list_item_names(
            user_id="alice",
            is_admin=False,
            created_date="2026-03-10",
        )

        assert [(item.kind, item.name) for item in result.items] == [
            (GalleryItemKind.VIDEO, "range-video.mp4"),
            (GalleryItemKind.IMAGE, "range-start.png"),
        ]
        assert result.total_count == 2


class TestGetBoardMediaSummaries:
    def test_returns_counts_and_deterministic_covers_in_one_result(self, services) -> None:
        populated = services["boards"].save("Populated", "alice")
        empty = services["boards"].save("Empty", "alice")
        _save_image(services["images"], "cover.png", user_id="alice")
        _save_video(services["videos"], "cover.mp4", user_id="alice")
        _save_video(services["videos"], "intermediate.mp4", user_id="alice")
        # An uploaded (user-category) video is an asset: counted in video_count AND
        # asset_video_count, so clients can split the Media/Assets views.
        _save_video(services["videos"], "uploaded.mp4", user_id="alice", category=ImageCategory.USER)
        services["board_images"].add_image_to_board(populated.board_id, "cover.png")
        services["board_videos"].add_video_to_board(populated.board_id, "cover.mp4")
        services["board_videos"].add_video_to_board(populated.board_id, "intermediate.mp4")
        services["board_videos"].add_video_to_board(populated.board_id, "uploaded.mp4")
        with services["images"]._db.transaction() as cursor:
            cursor.execute(
                "UPDATE images SET starred = 1, created_at = ? WHERE image_name = ?",
                ("2026-01-05 12:00:00", "cover.png"),
            )
            cursor.execute(
                "UPDATE videos SET starred = 1, created_at = ? WHERE video_name = ?",
                ("2026-01-05 12:00:00", "cover.mp4"),
            )
            cursor.execute(
                "UPDATE videos SET is_intermediate = 1 WHERE video_name = ?",
                ("intermediate.mp4",),
            )

        summaries, _, details = _capture_plan(
            services,
            lambda: services["gallery"].get_board_media_summaries([populated.board_id, empty.board_id]),
            "ROW_NUMBER() OVER",
        )

        assert summaries[populated.board_id].image_count == 1
        assert summaries[populated.board_id].video_count == 2
        assert summaries[populated.board_id].asset_count == 0
        assert summaries[populated.board_id].asset_video_count == 1
        assert summaries[populated.board_id].cover_image_name is None
        assert summaries[populated.board_id].cover_video_name == "cover.mp4"
        assert summaries[empty.board_id].image_count == 0
        assert summaries[empty.board_id].video_count == 0
        assert summaries[empty.board_id].asset_video_count == 0
        assert summaries[empty.board_id].cover_image_name is None
        assert summaries[empty.board_id].cover_video_name is None
        assert next(i for i, detail in enumerate(details) if "board_images" in detail) < next(
            i for i, detail in enumerate(details) if "images" in detail and "board_images" not in detail
        )
        assert next(i for i, detail in enumerate(details) if "board_videos" in detail) < next(
            i for i, detail in enumerate(details) if "videos" in detail and "board_videos" not in detail
        )


class TestGalleryQueryPlans:
    def test_default_name_list_omits_membership_join_and_query_hints(self, services) -> None:
        _save_image(services["images"], "image.png", user_id="alice")
        _save_video(services["videos"], "video.mp4", user_id="alice")

        result, statement, _ = _capture_plan(
            services,
            lambda: services["gallery"].list_item_names(
                categories=[ImageCategory.GENERAL],
                is_intermediate=False,
                is_admin=True,
            ),
            "UNION ALL",
        )

        assert {(item.kind, item.name) for item in result.items} == {
            (GalleryItemKind.IMAGE, "image.png"),
            (GalleryItemKind.VIDEO, "video.mp4"),
        }
        assert result.total_count == 2
        assert result.starred_count == 0
        assert "LEFT JOIN board_images" not in statement
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"search_term": "does-not-match"},
            {"user_id": "alice", "is_admin": False},
            {"user_id": "alice", "is_admin": False, "order_dir": SQLiteDirection.Ascending},
            {"user_id": "alice", "is_admin": False, "starred_first": False},
            {"user_id": "alice", "is_admin": False, "starred": True},
        ],
    )
    def test_name_shapes_do_not_force_indexes(self, services, kwargs) -> None:
        _save_image(services["images"], "image.png", user_id="alice")

        _, statement, _ = _capture_plan(
            services,
            lambda: services["gallery"].list_item_names(
                categories=[ImageCategory.GENERAL],
                is_intermediate=False,
                **kwargs,
            ),
            "UNION ALL",
        )

        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    def test_starred_filter_searches_the_starred_index(self, services) -> None:
        # The bounded starred strip asks for `starred = 1` on every board visit, so the
        # filter must be served by idx_*_starred rather than a table scan per half.
        _save_image(services["images"], "starred.png", user_id="alice")
        _save_image(services["images"], "plain.png", user_id="alice")
        _save_video(services["videos"], "starred.mp4", user_id="alice")
        _save_video(services["videos"], "plain.mp4", user_id="alice")
        _star(services, "images", "image_name", "starred.png")
        _star(services, "videos", "video_name", "starred.mp4")

        result, _, details = _capture_plan(
            services,
            lambda: services["gallery"].list_item_names(
                categories=[ImageCategory.GENERAL],
                is_intermediate=False,
                is_admin=True,
                starred=True,
            ),
            "UNION ALL",
        )

        assert result.total_count == 2
        assert not any(detail.startswith(("SCAN images", "SCAN videos")) for detail in details)

    def test_explicit_board_starts_from_mixed_membership(self, services) -> None:
        board = services["boards"].save("Small", "alice")
        _save_image(services["images"], "image.png", user_id="alice")
        _save_video(services["videos"], "video.mp4", user_id="alice")
        _save_image(services["images"], "outside.png", user_id="alice")
        services["board_images"].add_image_to_board(board.board_id, "image.png")
        services["board_videos"].add_video_to_board(board.board_id, "video.mp4")

        result, _, details = _capture_plan(
            services,
            lambda: services["gallery"].list_item_names(
                board_id=board.board_id,
                categories=[ImageCategory.GENERAL],
                is_intermediate=False,
            ),
            "UNION ALL",
        )

        assert {(item.kind, item.name) for item in result.items} == {
            (GalleryItemKind.IMAGE, "image.png"),
            (GalleryItemKind.VIDEO, "video.mp4"),
        }
        assert next(i for i, detail in enumerate(details) if "board_images" in detail) < next(
            i for i, detail in enumerate(details) if "images" in detail and "board_images" not in detail
        )
        assert next(i for i, detail in enumerate(details) if "board_videos" in detail) < next(
            i for i, detail in enumerate(details) if "videos" in detail and "board_videos" not in detail
        )

    def test_asset_category_keeps_mixed_result_shape(self, services) -> None:
        _save_image(services["images"], "asset.png", user_id="alice", category=ImageCategory.CONTROL)
        _save_video(services["videos"], "asset.mp4", user_id="alice", category=ImageCategory.CONTROL)
        _save_image(services["images"], "general.png", user_id="alice")

        result, statement, _ = _capture_plan(
            services,
            lambda: services["gallery"].list_item_names(
                categories=[ImageCategory.CONTROL],
                is_intermediate=False,
                is_admin=True,
            ),
            "UNION ALL",
        )

        assert {(item.kind, item.name) for item in result.items} == {
            (GalleryItemKind.IMAGE, "asset.png"),
            (GalleryItemKind.VIDEO, "asset.mp4"),
        }
        assert result.total_count == 2
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

        _, non_admin_statement, _ = _capture_plan(
            services,
            lambda: services["gallery"].list_item_names(
                categories=[ImageCategory.CONTROL],
                is_intermediate=False,
                user_id="alice",
                is_admin=False,
            ),
            "UNION ALL",
        )
        assert "INDEXED BY" not in non_admin_statement
        assert "NOT INDEXED" not in non_admin_statement

    def test_paginated_item_path_keeps_result_shape_and_avoids_gallery_index(self, services) -> None:
        _save_image(services["images"], "image.png", user_id="alice")
        _save_video(services["videos"], "video.mp4", user_id="alice")

        result, statement, _ = _capture_plan(
            services,
            lambda: services["gallery"].list_items(
                offset=1,
                limit=1,
                categories=[ImageCategory.GENERAL],
                is_intermediate=False,
                is_admin=True,
            ),
            "UNION ALL",
        )

        assert result.offset == 1
        assert result.limit == 1
        assert result.total == 2
        assert len(result.items) == 1
        assert result.items[0].kind in {GalleryItemKind.IMAGE, GalleryItemKind.VIDEO}
        assert result.items[0].full_url
        assert "LEFT JOIN board_images" in statement
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    def test_board_image_count_starts_from_membership(self, services) -> None:
        board = services["boards"].save("Small", "alice")
        _save_image(services["images"], "image.png", user_id="alice")
        services["board_images"].add_image_to_board(board.board_id, "image.png")

        count, _, details = _capture_plan(
            services,
            lambda: services["board_images"].get_image_count_for_board(board.board_id),
            "SELECT COUNT(*)",
        )

        assert count == 1
        assert next(i for i, detail in enumerate(details) if "board_images" in detail) < next(
            i for i, detail in enumerate(details) if "images" in detail and "board_images" not in detail
        )


class TestOrderingTieBreakers:
    """PR #9163 review fix: ordering only by (starred, created_at) left images and videos
    created within the same timestamp granularity with no defined relative order — rows
    could reorder across refetches or shift between offset pages, and the virtual-board
    cover could flicker between equally-new items."""

    SAME_TS = "2026-01-05 12:00:00"

    def _seed_same_timestamp(self, services) -> None:
        _save_image(services["images"], "b.png", user_id="alice")
        _save_image(services["images"], "a.png", user_id="alice")
        _save_video(services["videos"], "b.mp4", user_id="alice")
        _save_video(services["videos"], "a.mp4", user_id="alice")
        for table, col, name in [
            ("images", "image_name", "a.png"),
            ("images", "image_name", "b.png"),
            ("videos", "video_name", "a.mp4"),
            ("videos", "video_name", "b.mp4"),
        ]:
            _backdate(services, table, col, name, self.SAME_TS)

    def test_same_timestamp_order_is_deterministic(self, services) -> None:
        self._seed_same_timestamp(services)
        gallery = services["gallery"]

        first = [(i.kind, i.name) for i in gallery.list_item_names(user_id="alice", is_admin=False).items]
        for _ in range(5):
            again = [(i.kind, i.name) for i in gallery.list_item_names(user_id="alice", is_admin=False).items]
            assert again == first

        # Descending: videos sort before images ('video' > 'image'), names descending.
        assert first == [
            (GalleryItemKind.VIDEO, "b.mp4"),
            (GalleryItemKind.VIDEO, "a.mp4"),
            (GalleryItemKind.IMAGE, "b.png"),
            (GalleryItemKind.IMAGE, "a.png"),
        ]

    def test_ascending_is_mirror_of_descending(self, services) -> None:
        from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection

        self._seed_same_timestamp(services)
        gallery = services["gallery"]

        desc = [(i.kind, i.name) for i in gallery.list_item_names(user_id="alice", is_admin=False).items]
        asc = [
            (i.kind, i.name)
            for i in gallery.list_item_names(user_id="alice", is_admin=False, order_dir=SQLiteDirection.Ascending).items
        ]
        assert asc == list(reversed(desc))

    def test_same_timestamp_cover_is_deterministic(self, services) -> None:
        self._seed_same_timestamp(services)
        gallery = services["gallery"]

        covers = set()
        for _ in range(5):
            boards = gallery.get_dates(user_id="alice", is_admin=False)
            assert len(boards) == 1
            covers.add((boards[0].cover_image_name, boards[0].cover_video_name))

        # One stable choice across refetches: the kind/name-descending winner (b.mp4).
        assert covers == {(None, "b.mp4")}


def _save_marked_video(store: SqliteVideoRecordStorage, name: str, user_id: str, metadata: str | None) -> None:
    store.save(
        video_name=name,
        video_origin=ResourceOrigin.EXTERNAL,
        video_category=ImageCategory.USER,
        width=640,
        height=360,
        duration=1.0,
        fps=24.0,
        has_workflow=False,
        is_intermediate=False,
        metadata=metadata,
        user_id=user_id,
    )


class TestMediaOriginOnListedItems:
    """The listing carries `media_origin`, so a clip picked straight off the gallery grid
    can be conditioned correctly without a follow-up /metadata request.

    "Extend in Video" builds its source clip from a listed item rather than a resolve, so
    the marker has to survive the polymorphic UNION as well as the video DTO.
    """

    def test_a_wrapped_audio_upload_is_marked(self, services) -> None:
        _save_marked_video(services["videos"], "wrapped.mp4", "alice", '{"media_origin": "audio_upload"}')

        listed = services["gallery"].list_items(user_id="alice", is_admin=False)

        assert [(item.name, item.media_origin) for item in listed.items] == [("wrapped.mp4", "audio_upload")]

    def test_ordinary_videos_and_images_carry_no_marker(self, services) -> None:
        _save_marked_video(services["videos"], "plain.mp4", "alice", '{"note": "kept"}')
        _save_marked_video(services["videos"], "bare.mp4", "alice", None)
        _save_image(services["images"], "still.png", user_id="alice")

        listed = services["gallery"].list_items(user_id="alice", is_admin=False)
        origins = {item.name: item.media_origin for item in listed.items}

        assert origins == {"plain.mp4": None, "bare.mp4": None, "still.png": None}

    def test_a_non_string_marker_does_not_break_the_listing(self, services) -> None:
        """One video with an odd `media_origin` must not fail the whole gallery page.

        Upload metadata is validated only as a JSON object, so the extracted value can be an
        int; building `GalleryItem` from it used to raise, and the listing is a UNION over
        every item, so the failure was not confined to the offending video.
        """
        _save_marked_video(services["videos"], "odd.mp4", "alice", '{"media_origin": 7}')
        _save_marked_video(services["videos"], "wrapped.mp4", "alice", '{"media_origin": "audio_upload"}')

        listed = services["gallery"].list_items(user_id="alice", is_admin=False)
        origins = {item.name: item.media_origin for item in listed.items}

        assert origins == {"odd.mp4": None, "wrapped.mp4": "audio_upload"}

    def test_a_malformed_metadata_blob_does_not_fail_the_page(self, services) -> None:
        """The listing is a UNION over every item, so an unguarded `json_extract` raise here
        would 500 the whole gallery page for one bad row — and for an admin, for everyone."""
        _save_marked_video(services["videos"], "bad.mp4", "alice", "not json at all")
        _save_marked_video(services["videos"], "wrapped.mp4", "alice", '{"media_origin": "audio_upload"}')

        listed = services["gallery"].list_items(user_id="alice", is_admin=False)

        assert {item.name: item.media_origin for item in listed.items} == {
            "bad.mp4": None,
            "wrapped.mp4": "audio_upload",
        }
