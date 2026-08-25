"""Regression tests for SqliteVideoRecordStorage multiuser isolation.

Covers JPPhoto's code-review finding (PR #9163): when ``board_id`` was omitted
from /v1/videos/ and /v1/videos/names, the SQL builder applied no user filter
and a non-admin caller saw every user's videos. The fix added an
``elif user_id is not None and not is_admin`` branch; these tests pin the
behaviour so the regression cannot reappear.
"""

import sqlite3

import pytest

from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException
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


class TestDeterministicOrdering:
    @staticmethod
    def _set_same_timestamp(store: SqliteVideoRecordStorage) -> None:
        with store._db.transaction() as cursor:
            cursor.execute("UPDATE videos SET created_at = '2026-07-20 00:00:00.000'")

    def test_same_timestamp_pages_have_stable_mirrored_order(self, seeded_store: SqliteVideoRecordStorage) -> None:
        self._set_same_timestamp(seeded_store)

        ascending = [
            seeded_store.get_many(
                offset=offset,
                limit=1,
                starred_first=False,
                order_dir=SQLiteDirection.Ascending,
                user_id="alice",
            )
            .items[0]
            .video_name
            for offset in range(2)
        ]
        descending = [
            seeded_store.get_many(
                offset=offset,
                limit=1,
                starred_first=False,
                order_dir=SQLiteDirection.Descending,
                user_id="alice",
            )
            .items[0]
            .video_name
            for offset in range(2)
        ]

        assert ascending == ["alice_1.mp4", "alice_2.mp4"]
        assert descending == list(reversed(ascending))

    def test_same_timestamp_name_list_has_stable_mirrored_order(self, seeded_store: SqliteVideoRecordStorage) -> None:
        self._set_same_timestamp(seeded_store)

        ascending = seeded_store.get_video_names(
            starred_first=False,
            order_dir=SQLiteDirection.Ascending,
            user_id="alice",
        ).video_names
        descending = seeded_store.get_video_names(
            starred_first=False,
            order_dir=SQLiteDirection.Descending,
            user_id="alice",
        ).video_names

        assert ascending == ["alice_1.mp4", "alice_2.mp4"]
        assert descending == list(reversed(ascending))

    def test_board_cover_uses_name_as_same_timestamp_tie_breaker(self, store: SqliteVideoRecordStorage) -> None:
        board = SqliteBoardRecordStorage(store._db).save("Board", "system")
        _save(store, "a.mp4", user_id="system")
        _save(store, "b.mp4", user_id="system")
        self._set_same_timestamp(store)
        with store._db.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO board_videos (board_id, video_name) VALUES (?, ?)",
                [(board.board_id, "a.mp4"), (board.board_id, "b.mp4")],
            )

        assert store.get_most_recent_video_for_board(board.board_id).video_name == "b.mp4"


class TestUserDeletionLifecycle:
    """Documents the intended videos↔users lifecycle (JPPhoto PR #9163 July-10 follow-up).

    ``videos.user_id`` deliberately has no FK to ``users`` — exactly like ``images``,
    ``boards`` and ``workflows``, whose user_id columns (migration_27) are index-only.
    Deleting a user therefore leaves their videos in place instead of cascading a row
    delete that would strand the files on disk; the orphaned records stay visible to
    administrators (and only to administrators), who can clean them up or reassign them.
    These tests pin that platform-wide behavior for videos so any future change to the
    user-deletion story is a deliberate decision rather than an accident.
    """

    @pytest.fixture
    def migrated_db(self) -> SqliteDatabase:
        config = InvokeAIAppConfig(use_memory_db=True)
        logger = InvokeAILogger.get_logger(config=config)
        return create_mock_sqlite_database(config, logger)

    def test_videos_survive_owner_deletion_and_remain_admin_only(self, migrated_db: SqliteDatabase) -> None:
        users = UserService(migrated_db)
        store = SqliteVideoRecordStorage(db=migrated_db)

        owner = users.create(
            UserCreateRequest(
                email="doomed@example.com",
                display_name="Doomed User",
                password="TestPassword123",
                is_admin=False,
            )
        )
        _save(store, "doomed.mp4", user_id=owner.user_id)

        users.delete(owner.user_id)
        assert users.get(owner.user_id) is None

        # The record survives, still attributed to the deleted owner...
        assert store.get_user_id("doomed.mp4") == owner.user_id
        # ...is visible to admins for cleanup...
        admin_view = store.get_many(user_id="some-admin", is_admin=True)
        assert "doomed.mp4" in {v.video_name for v in admin_view.items}
        # ...and no regular user inherits it.
        other_view = store.get_many(user_id="bystander", is_admin=False)
        assert "doomed.mp4" not in {v.video_name for v in other_view.items}


def test_get_propagates_a_storage_error_instead_of_reporting_the_row_missing(store: SqliteVideoRecordStorage):
    """An unreadable database must not be indistinguishable from a deleted video.

    `get` used to translate every sqlite3.Error into VideoRecordNotFoundException, which made
    that exception mean "the row is absent, OR the read failed". Two callers act destructively
    on it: `_assert_video_read_access` answers 404 for a positive not-found and the clients drop
    their reference to the video on one, and the staged-delete recovery reads it as proof the
    delete committed and purges the staged files.
    """
    _save(store, "video-1.mp4", "user-1")
    # A real storage failure rather than a patched one: the SELECT below cannot run at all, the
    # same shape a locked or corrupt database presents. The row's absence is not what is being
    # reported, and the caller must be able to tell.
    with store._db.transaction() as cursor:
        cursor.execute("DROP TABLE videos;")

    with pytest.raises(sqlite3.OperationalError):
        store.get("video-1.mp4")


def test_get_still_reports_a_positively_absent_row_as_missing(store: SqliteVideoRecordStorage):
    """The narrowing stays exactly that narrow: absence is still absence."""
    with pytest.raises(VideoRecordNotFoundException):
        store.get("never-existed.mp4")


def test_exists_reports_a_row_get_cannot_deserialize(store: SqliteVideoRecordStorage):
    """Presence, not readability. `get` would raise on an enum value this version does not know
    — a row written by a newer one — and the refusal path reads that as absence, which would
    report a live video gone."""
    _save(store, "video-1.mp4", "user-1")
    with store._db.transaction() as cursor:
        cursor.execute("UPDATE videos SET video_category = 'from_the_future' WHERE video_name = ?;", ("video-1.mp4",))

    with pytest.raises(ValueError):
        store.get("video-1.mp4")
    assert store.exists("video-1.mp4") is True


def test_exists_propagates_a_storage_error(store: SqliteVideoRecordStorage):
    """ "Could not look" is not "not there" — the caller answers 404 on a False."""
    with store._db.transaction() as cursor:
        cursor.execute("DROP TABLE videos;")

    with pytest.raises(sqlite3.OperationalError):
        store.exists("video-1.mp4")
