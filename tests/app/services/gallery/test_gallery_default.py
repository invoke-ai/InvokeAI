"""Regression tests for the gallery service's multiuser isolation and date-based
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
from typing import Any, TypeVar

import pytest
from sqlalchemy import update

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
from invokeai.app.services.gallery.gallery_common import GalleryItemKind
from invokeai.app.services.gallery.gallery_default import GalleryService
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import gallery as gallery_queries
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from tests.fixtures.database import capture_statements, explain_query_plan


@pytest.fixture
def services(database: Database):
    gallery = GalleryService(database)
    gallery.start(SimpleNamespace(services=SimpleNamespace(urls=LocalUrlService())))  # type: ignore[arg-type]
    return {
        "db": database,
        "gallery": gallery,
        "images": ImageRecordStorage(database),
        "videos": VideoRecordStorage(database),
        "boards": BoardRecordStorage(database),
        "board_images": BoardImageRecordStorage(database),
        "board_videos": BoardVideoRecordStorage(database),
    }


def _save_image(
    store: ImageRecordStorage,
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
    store: VideoRecordStorage,
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
    database = services["db"]
    with capture_statements(database) as statements:
        result = call()
    statement, parameters = next((sql, parameters) for sql, parameters in statements if statement_marker in sql)
    return result, statement, explain_query_plan(database, statement, parameters)


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
    _update(services, table, name_col, name, created_at=created_at)


def _star(services, table: str, name_col: str, name: str) -> None:
    _update(services, table, name_col, name, starred=True)


def _update(services, table: str, name_col: str, name: str, **values: object) -> None:
    media = {"images": images, "videos": videos}[table]
    with services["db"].begin(write=True) as conn:
        conn.execute(update(media).where(media.c[name_col] == name).values(**values))


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
        _update(services, "images", "image_name", "cover.png", starred=True, created_at="2026-01-05 12:00:00")
        _update(services, "videos", "video_name", "cover.mp4", starred=True, created_at="2026-01-05 12:00:00")
        _update(services, "videos", "video_name", "intermediate.mp4", is_intermediate=True)

        summaries = services["gallery"].get_board_media_summaries([populated.board_id, empty.board_id])

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

    @pytest.mark.sqlite_only
    def test_summaries_start_from_the_boards_memberships(self, services) -> None:
        board = services["boards"].save("Small", "alice")
        _save_image(services["images"], "image.png", user_id="alice")
        _save_video(services["videos"], "video.mp4", user_id="alice")
        services["board_images"].add_image_to_board(board.board_id, "image.png")
        services["board_videos"].add_video_to_board(board.board_id, "video.mp4")

        _, _, details = _capture_plan(
            services, lambda: services["gallery"].get_board_media_summaries([board.board_id]), "row_number() OVER"
        )

        assert next(i for i, detail in enumerate(details) if "board_images" in detail) < next(
            i for i, detail in enumerate(details) if "images" in detail and "board_images" not in detail
        )
        assert next(i for i, detail in enumerate(details) if "board_videos" in detail) < next(
            i for i, detail in enumerate(details) if "videos" in detail and "board_videos" not in detail
        )


@pytest.mark.sqlite_only
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
        assert "JOIN board_images" not in statement
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
        assert "LEFT OUTER JOIN board_images" in statement
        assert "INDEXED BY" not in statement
        assert "NOT INDEXED" not in statement

    def test_board_image_count_starts_from_membership(self, services) -> None:
        board = services["boards"].save("Small", "alice")
        _save_image(services["images"], "image.png", user_id="alice")
        services["board_images"].add_image_to_board(board.board_id, "image.png")

        counts, _, details = _capture_plan(
            services,
            lambda: services["board_images"].get_counts_for_board(board.board_id),
            "FROM board_images CROSS JOIN images",
        )

        assert counts == (1, 0)
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


def _save_marked_video(store: VideoRecordStorage, name: str, user_id: str, metadata: str | None) -> None:
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


class TestCanvasOwnedImagesAreNotCounted:
    """OTHER is the category of images a canvas layer owns: neither the images view nor the assets view lists them,
    so neither a day's nor a board's counts may include them, or a badge disagrees with what it opens."""

    def test_a_day_counts_only_what_its_views_list(self, services) -> None:
        for name, category in (
            ("generated.png", ImageCategory.GENERAL),
            ("uploaded.png", ImageCategory.USER),
            ("painted.png", ImageCategory.OTHER),
        ):
            _save_image(services["images"], name, user_id="alice", category=category)
            _backdate(services, "images", "image_name", name, "2026-01-05 12:00:00.000")

        (day,) = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert (day.image_count, day.asset_count) == (1, 1)

    def test_a_board_counts_only_what_its_views_list(self, services) -> None:
        board = services["boards"].save("Project", "alice")
        for name, category in (
            ("generated.png", ImageCategory.GENERAL),
            ("uploaded.png", ImageCategory.USER),
            ("painted.png", ImageCategory.OTHER),
        ):
            _save_image(services["images"], name, user_id="alice", category=category)
            services["board_images"].add_image_to_board(board.board_id, name)

        summary = services["gallery"].get_board_media_summaries([board.board_id])[board.board_id]

        assert (summary.image_count, summary.asset_count) == (1, 1)
        # The same counts as the board's own badge.
        assert services["board_images"].get_counts_for_board(board.board_id) == (1, 1)


class TestFilters:
    def test_search_ignores_case_and_matches_percent_and_underscore_literally(self, services) -> None:
        for name, prompt in (("hit.png", "a 100% Cat_Photo"), ("near.png", "a 100x catXphoto")):
            services["images"].save(
                image_name=name,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                width=8,
                height=8,
                has_workflow=False,
                metadata=f'{{"prompt": "{prompt}"}}',
                user_id="alice",
            )

        result = services["gallery"].list_item_names(search_term="100% cat_", is_admin=True)

        assert [item.name for item in result.items] == ["hit.png"]

    def test_a_search_ignores_the_case_of_letters_beyond_ascii(self, services) -> None:
        services["images"].save(
            image_name="apples.png",
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=8,
            height=8,
            has_workflow=False,
            metadata='{"prompt": "frische äpfel"}',
            user_id="alice",
        )

        result = services["gallery"].list_item_names(search_term="ÄPFEL", is_admin=True)

        assert [item.name for item in result.items] == ["apples.png"]

    def test_the_categories_filter_applies_to_both_kinds_in_pages_and_names(self, services) -> None:
        for name, category in (("general.png", ImageCategory.GENERAL), ("control.png", ImageCategory.CONTROL)):
            _save_image(services["images"], name, user_id="alice", category=category)
        for name, category in (("general.mp4", ImageCategory.GENERAL), ("user.mp4", ImageCategory.USER)):
            _save_video(services["videos"], name, user_id="alice", category=category)
        assets = [ImageCategory.CONTROL, ImageCategory.USER]

        names = services["gallery"].list_item_names(categories=assets, is_admin=True)
        page = services["gallery"].list_items(limit=10, categories=assets, is_admin=True)

        assert {item.name for item in names.items} == {"control.png", "user.mp4"}
        assert {item.name for item in page.items} == {"control.png", "user.mp4"} and page.total == 2

    def test_the_origin_filter_applies_to_both_kinds(self, services) -> None:
        _save_image(services["images"], "internal.png", user_id="alice")
        _save_video(services["videos"], "internal.mp4", user_id="alice")
        services["videos"].save(
            video_name="external.mp4",
            video_origin=ResourceOrigin.EXTERNAL,
            video_category=ImageCategory.GENERAL,
            width=8,
            height=8,
            duration=1.0,
            fps=8.0,
            has_workflow=False,
            user_id="alice",
        )

        result = services["gallery"].list_items(limit=10, origin=ResourceOrigin.EXTERNAL, is_admin=True)

        assert [item.name for item in result.items] == ["external.mp4"] and result.total == 1

    @pytest.mark.parametrize("day", ["2026-1-05", "not a day", "2026-01-05 00:00", "9999-12-31", ""])
    def test_a_day_that_is_no_iso_day_or_the_last_one_matches_nothing(self, services, day: str) -> None:
        _save_image(services["images"], "image.png", user_id="alice")
        _backdate(services, "images", "image_name", "image.png", "2026-01-05 12:00:00.000")

        assert services["gallery"].list_item_names(created_date=day, is_admin=True).items == []
        assert services["gallery"].list_items(limit=10, created_to=day, is_admin=True).total == 0

    def test_pages_walk_the_listing_and_a_page_of_none_still_counts(self, services) -> None:
        for index, name in enumerate(("a.png", "b.png")):
            _save_image(services["images"], name, user_id="alice")
            _backdate(services, "images", "image_name", name, f"2026-01-0{index + 1} 00:00:00.000")
        _save_video(services["videos"], "c.mp4", user_id="alice")
        _backdate(services, "videos", "video_name", "c.mp4", "2026-01-03 00:00:00.000")
        names = [item.name for item in services["gallery"].list_item_names(is_admin=True).items]

        pages = [services["gallery"].list_items(offset=offset, limit=1, is_admin=True) for offset in range(3)]
        count_only = services["gallery"].list_items(limit=0, is_admin=True)

        assert [[item.name for item in page.items] for page in pages] == [[name] for name in names]
        assert {page.total for page in pages} == {3}
        assert count_only.items == [] and count_only.total == 3

    def test_a_board_lists_only_its_own_items(self, services) -> None:
        mine, other = services["boards"].save("Mine", "alice"), services["boards"].save("Other", "alice")
        for name, board in (("mine.png", mine), ("other.png", other)):
            _save_image(services["images"], name, user_id="alice")
            services["board_images"].add_image_to_board(board.board_id, name)
        for name, board in (("mine.mp4", mine), ("other.mp4", other)):
            _save_video(services["videos"], name, user_id="alice")
            services["board_videos"].add_video_to_board(board.board_id, name)

        names = services["gallery"].list_item_names(board_id=mine.board_id, is_admin=True)
        page = services["gallery"].list_items(limit=10, board_id=mine.board_id, is_admin=True)

        assert {item.name for item in names.items} == {"mine.png", "mine.mp4"}
        assert {item.name for item in page.items} == {"mine.png", "mine.mp4"} and page.total == 2
        assert {item.board_id for item in page.items} == {mine.board_id}


class TestReviewedBehaviour:
    """Pages, names, dates and summaries in the cases the review of the port found unpinned."""

    def test_shared_board_lists_owners_items_for_viewer(self, services) -> None:
        board = services["boards"].save("Bob's", "bob")
        services["boards"].update(board.board_id, BoardChanges(board_visibility=BoardVisibility.Shared))
        _save_image(services["images"], "bob.png", user_id="bob")
        _save_video(services["videos"], "bob.mp4", user_id="bob")
        services["board_images"].add_image_to_board(board.board_id, "bob.png")
        services["board_videos"].add_video_to_board(board.board_id, "bob.mp4")
        _start_gallery_for_item_results(services)
        g = services["gallery"]

        names = g.get_item_names(board_id=board.board_id, user_id="alice", is_admin=False)
        page = g.list_items(limit=10, board_id=board.board_id, user_id="alice", is_admin=False)

        assert set(names.item_names) == {"bob.png", "bob.mp4"}
        assert {i.name for i in page.items} == {"bob.png", "bob.mp4"} and page.total == 2

    def test_intermediates_are_left_out_of_pages_names_and_totals(self, services) -> None:
        for name in ("kept.png", "temp.png"):
            _save_image(services["images"], name, user_id="alice")
        for name in ("kept.mp4", "temp.mp4"):
            _save_video(services["videos"], name, user_id="alice")
        _update(services, "images", "image_name", "temp.png", is_intermediate=True)
        _update(services, "videos", "video_name", "temp.mp4", is_intermediate=True)
        _start_gallery_for_item_results(services)
        g = services["gallery"]

        names = g.get_item_names(is_intermediate=False, user_id="alice", is_admin=False)
        page = g.list_items(limit=10, is_intermediate=False, user_id="alice", is_admin=False)
        only = g.get_item_names(is_intermediate=True, user_id="alice", is_admin=False)

        assert set(names.item_names) == {"kept.png", "kept.mp4"}
        assert {i.name for i in page.items} == {"kept.png", "kept.mp4"} and page.total == 2
        assert set(only.item_names) == {"temp.png", "temp.mp4"}

    def test_pages_follow_direction_and_starred_first_like_names(self, services) -> None:
        for name in ("a.png", "b.png"):
            _save_image(services["images"], name, user_id="alice")
        for name in ("a.mp4", "b.mp4"):
            _save_video(services["videos"], name, user_id="alice")
        _save_image(services["images"], "old-starred.png", user_id="alice")
        for table, col, name in [
            ("images", "image_name", "a.png"),
            ("images", "image_name", "b.png"),
            ("videos", "video_name", "a.mp4"),
            ("videos", "video_name", "b.mp4"),
        ]:
            _backdate(services, table, col, name, "2026-01-05 12:00:00.000")
        _backdate(services, "images", "image_name", "old-starred.png", "2026-01-01 12:00:00.000")
        _star(services, "images", "image_name", "old-starred.png")
        _start_gallery_for_item_results(services)
        g = services["gallery"]

        for order_dir in (SQLiteDirection.Ascending, SQLiteDirection.Descending):
            for starred_first in (True, False):
                kwargs: dict[str, Any] = {
                    "order_dir": order_dir,
                    "starred_first": starred_first,
                    "user_id": "alice",
                    "is_admin": False,
                }
                expected = g.get_item_names(**kwargs).item_names
                page = [i.name for i in g.list_items(limit=10, **kwargs).items]
                walked = [g.list_items(offset=o, limit=2, **kwargs).items for o in (0, 2, 4)]
                assert page == expected, (order_dir, starred_first)
                assert [i.name for p in walked for i in p] == expected, (order_dir, starred_first)
        assert (
            g.get_item_names(order_dir=SQLiteDirection.Ascending, starred_first=False, user_id="alice").item_names[0]
            == "old-starred.png"
        )

    def test_a_days_cover_is_the_viewers_own(self, services) -> None:
        _save_image(services["images"], "alice.png", user_id="alice")
        _save_image(services["images"], "bob.png", user_id="bob")
        _backdate(services, "images", "image_name", "alice.png", "2026-01-05 09:00:00")
        _backdate(services, "images", "image_name", "bob.png", "2026-01-05 10:00:00")

        (day,) = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert day.cover_image_name == "alice.png"

    def test_intermediates_neither_make_nor_count_nor_cover_a_day(self, services) -> None:
        _save_image(services["images"], "kept.png", user_id="alice")
        _save_video(services["videos"], "temp.mp4", user_id="alice")
        _save_image(services["images"], "lone-temp.png", user_id="alice")
        _backdate(services, "images", "image_name", "kept.png", "2026-01-05 09:00:00")
        _backdate(services, "videos", "video_name", "temp.mp4", "2026-01-05 10:00:00")
        _backdate(services, "images", "image_name", "lone-temp.png", "2026-01-06 10:00:00")
        _update(services, "videos", "video_name", "temp.mp4", is_intermediate=True)
        _update(services, "images", "image_name", "lone-temp.png", is_intermediate=True)

        days = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert [(d.date, d.image_count, d.video_count, d.cover_image_name) for d in days] == [
            ("2026-01-05", 1, 0, "kept.png")
        ]

    def test_unboarded_pages_and_totals_are_the_viewers_own(self, services) -> None:
        _save_image(services["images"], "alice.png", user_id="alice")
        _save_video(services["videos"], "bob.mp4", user_id="bob")
        _start_gallery_for_item_results(services)

        page = services["gallery"].list_items(limit=10, board_id="none", user_id="alice", is_admin=False)

        assert [i.name for i in page.items] == ["alice.png"] and page.total == 1

    def test_page_items_carry_their_columns(self, services) -> None:
        board = services["boards"].save("B", "alice")
        services["videos"].save(
            video_name="clip.mp4",
            video_origin=ResourceOrigin.INTERNAL,
            video_category=ImageCategory.GENERAL,
            width=640,
            height=360,
            duration=2.5,
            fps=24.0,
            has_workflow=False,
            is_intermediate=False,
            user_id="alice",
        )
        services["board_videos"].add_video_to_board(board.board_id, "clip.mp4")
        _start_gallery_for_item_results(services)

        (item,) = services["gallery"].list_items(limit=10, user_id="alice", is_admin=False).items

        assert (item.width, item.height, item.duration, item.fps, item.board_id) == (
            640,
            360,
            2.5,
            24.0,
            board.board_id,
        )

    def test_board_summaries_span_chunks(self, services, monkeypatch) -> None:
        monkeypatch.setattr(gallery_queries, "IN_CHUNK", 2)
        boards = [services["boards"].save(f"B{i}", "alice") for i in range(3)]
        for i, board in enumerate(boards):
            _save_image(services["images"], f"{i}.png", user_id="alice")
            services["board_images"].add_image_to_board(board.board_id, f"{i}.png")

        summaries = services["gallery"].get_board_media_summaries([b.board_id for b in boards])

        assert [summaries[b.board_id].cover_image_name for b in boards] == ["0.png", "1.png", "2.png"]

    def test_board_cover_is_the_newest_without_stars(self, services) -> None:
        board = services["boards"].save("B", "alice")
        for name in ("z-old.png", "a-new.png"):
            _save_image(services["images"], name, user_id="alice")
            services["board_images"].add_image_to_board(board.board_id, name)
        _backdate(services, "images", "image_name", "z-old.png", "2026-01-01 00:00:00")
        _backdate(services, "images", "image_name", "a-new.png", "2026-01-02 00:00:00")

        summary = services["gallery"].get_board_media_summaries([board.board_id])[board.board_id]

        assert summary.cover_image_name == "a-new.png"

    def test_canvas_owned_image_neither_makes_nor_covers_a_day(self, services) -> None:
        _save_image(services["images"], "generated.png", user_id="alice")
        _save_image(services["images"], "painted.png", user_id="alice", category=ImageCategory.OTHER)
        _save_image(services["images"], "lone-painted.png", user_id="alice", category=ImageCategory.OTHER)
        _backdate(services, "images", "image_name", "generated.png", "2026-01-05 09:00:00")
        _backdate(services, "images", "image_name", "painted.png", "2026-01-05 10:00:00")
        _backdate(services, "images", "image_name", "lone-painted.png", "2026-01-06 10:00:00")

        days = services["gallery"].get_dates(user_id="alice", is_admin=False)

        assert [(d.date, d.cover_image_name) for d in days] == [("2026-01-05", "generated.png")]
