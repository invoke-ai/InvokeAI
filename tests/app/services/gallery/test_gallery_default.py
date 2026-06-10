"""Regression tests for SqliteGalleryService multiuser isolation.

Covers JPPhoto's code-review finding (PR #9163): the gallery /items/ and
/items/names endpoints returned every user's items when ``board_id`` was
omitted, because ``_build_half`` only applied a user filter for the explicit
"none" sentinel. The fix added an ``elif user_id is not None and not is_admin``
branch; these tests pin the behaviour for both halves of the polymorphic union.
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
