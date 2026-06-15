"""Regression tests for SqliteVideoRecordStorage multiuser isolation.

Covers JPPhoto's code-review finding (PR #9163): when ``board_id`` was omitted
from /v1/videos/ and /v1/videos/names, the SQL builder applied no user filter
and a non-admin caller saw every user's videos. The fix added an
``elif user_id is not None and not is_admin`` branch; these tests pin the
behaviour so the regression cannot reappear.
"""

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def store() -> SqliteVideoRecordStorage:
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return SqliteVideoRecordStorage(db=db)


def _save(store: SqliteVideoRecordStorage, name: str, user_id: str) -> None:
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
def seeded_store(store: SqliteVideoRecordStorage) -> SqliteVideoRecordStorage:
    # Two videos per user; all without board association (the bug occurred when board_id
    # was omitted from the query).
    _save(store, "alice_1.mp4", user_id="alice")
    _save(store, "alice_2.mp4", user_id="alice")
    _save(store, "bob_1.mp4", user_id="bob")
    _save(store, "bob_2.mp4", user_id="bob")
    return store


class TestGetManyOmittedBoardIdMultiuser:
    """get_many() with board_id=None must filter by user_id for non-admin callers."""

    def test_non_admin_only_sees_own_videos(self, seeded_store: SqliteVideoRecordStorage) -> None:
        result = seeded_store.get_many(user_id="alice", is_admin=False)
        names = {v.video_name for v in result.items}
        assert names == {"alice_1.mp4", "alice_2.mp4"}
        assert result.total == 2

    def test_admin_sees_every_users_videos(self, seeded_store: SqliteVideoRecordStorage) -> None:
        result = seeded_store.get_many(user_id="alice", is_admin=True)
        names = {v.video_name for v in result.items}
        assert names == {"alice_1.mp4", "alice_2.mp4", "bob_1.mp4", "bob_2.mp4"}

    def test_no_user_id_returns_all(self, seeded_store: SqliteVideoRecordStorage) -> None:
        # No user_id means the caller is bypassing user filtering entirely (e.g. internal calls).
        result = seeded_store.get_many(user_id=None, is_admin=False)
        names = {v.video_name for v in result.items}
        assert names == {"alice_1.mp4", "alice_2.mp4", "bob_1.mp4", "bob_2.mp4"}


class TestGetVideoNamesOmittedBoardIdMultiuser:
    """get_video_names() with board_id=None must filter by user_id for non-admin callers."""

    def test_non_admin_only_sees_own_videos(self, seeded_store: SqliteVideoRecordStorage) -> None:
        result = seeded_store.get_video_names(user_id="alice", is_admin=False)
        assert set(result.video_names) == {"alice_1.mp4", "alice_2.mp4"}
        assert result.total_count == 2

    def test_admin_sees_every_users_videos(self, seeded_store: SqliteVideoRecordStorage) -> None:
        result = seeded_store.get_video_names(user_id="alice", is_admin=True)
        assert set(result.video_names) == {"alice_1.mp4", "alice_2.mp4", "bob_1.mp4", "bob_2.mp4"}

    def test_explicit_none_board_still_isolates(self, seeded_store: SqliteVideoRecordStorage) -> None:
        # The "none" sentinel (uncategorized) must also apply the user filter — this was the
        # only path that was correct *before* the fix; the test guards against accidental
        # regression there too.
        result = seeded_store.get_video_names(board_id="none", user_id="alice", is_admin=False)
        assert set(result.video_names) == {"alice_1.mp4", "alice_2.mp4"}
