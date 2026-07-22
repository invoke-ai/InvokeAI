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

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.gallery.gallery_common import GalleryItemKind
from invokeai.app.services.gallery.gallery_default import SqliteGalleryService
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def services():
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return {
        "gallery": SqliteGalleryService(db=db),
        "images": SqliteImageRecordStorage(db=db),
        "videos": SqliteVideoRecordStorage(db=db),
    }


def _save_image(store: SqliteImageRecordStorage, name: str, user_id: str) -> None:
    store.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


def _save_video(store: SqliteVideoRecordStorage, name: str, user_id: str) -> None:
    store.save(
        video_name=name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


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
