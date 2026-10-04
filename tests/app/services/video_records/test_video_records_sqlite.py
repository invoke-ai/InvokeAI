"""Regression tests for SqliteVideoRecordStorage multiuser isolation.

Covers JPPhoto's code-review finding (PR #9163): when ``board_id`` was omitted
from /v1/videos/ and /v1/videos/names, the SQL builder applied no user filter
and a non-admin caller saw every user's videos. The fix added an
``elif user_id is not None and not is_admin`` branch; these tests pin the
behaviour so the regression cannot reappear.
"""

import json
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


def _save_with_metadata(store: SqliteVideoRecordStorage, name: str, metadata: str | None) -> None:
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
        user_id="alice",
    )


def test_media_origin_is_projected_out_of_the_metadata_blob(store: SqliteVideoRecordStorage) -> None:
    """The record carries `media_origin` so clients need no separate /metadata request.

    An audio upload the ingest converter wrapped into a waveform video is marked
    `audio_upload`; the frontend starts such a reference on its soundtrack alone, and a
    wrong value there costs a video reference slot and roughly doubles the packed sequence.
    """
    _save_with_metadata(store, "wrapped.mp4", '{"media_origin": "audio_upload", "note": "kept"}')

    assert store.get("wrapped.mp4").media_origin == "audio_upload"
    # The rest of the blob stays behind the /metadata route rather than riding every row.
    assert store.get_metadata("wrapped.mp4") is not None


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="no metadata at all"),
        pytest.param("{}", id="metadata without the key"),
        pytest.param('{"note": "kept"}', id="metadata with other keys"),
    ],
)
def test_media_origin_is_none_when_unmarked(store: SqliteVideoRecordStorage, metadata: str | None) -> None:
    """`json_extract` over a NULL or key-less blob yields NULL rather than raising."""
    _save_with_metadata(store, "plain.mp4", metadata)

    assert store.get("plain.mp4").media_origin is None


def test_media_origin_survives_a_listing(store: SqliteVideoRecordStorage) -> None:
    """`get_many` selects the same columns, so a listed row carries the marker too."""
    _save_with_metadata(store, "wrapped.mp4", '{"media_origin": "audio_upload"}')
    _save_with_metadata(store, "plain.mp4", None)

    listed = store.get_many(offset=0, limit=10, order_dir=SQLiteDirection.Descending, user_id="alice", is_admin=False)
    origins = {record.video_name: record.media_origin for record in listed.items}

    assert origins == {"wrapped.mp4": "audio_upload", "plain.mp4": None}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("1", id="a JSON number"),
        pytest.param("true", id="a JSON boolean"),
        pytest.param("1.5", id="a JSON float"),
        pytest.param('""', id="an empty string, which the frontend also reads as no marker"),
        # `json_extract` hands an object or array back as its SERIALIZED TEXT, so these
        # arrive as `str` and an isinstance check alone would propagate them.
        pytest.param('{"a": "b"}', id="a JSON object, returned as its text"),
        pytest.param("[1, 2]", id="a JSON array, returned as its text"),
        pytest.param('"has a space"', id="a string that is not marker-shaped"),
    ],
)
def test_a_non_string_marker_does_not_break_the_record(store: SqliteVideoRecordStorage, raw: str) -> None:
    """A client may store any JSON value under `media_origin`; the row must still deserialize.

    `MetadataField` validates only that the blob is an object, so `json_extract` can hand
    back an int or a float. Passing one to the `Optional[str]` field raised while building
    the record -- which took out the video's DTO *and* every gallery listing containing it,
    so one odd upload broke the gallery.
    """
    _save_with_metadata(store, "odd.mp4", '{"media_origin": ' + raw + "}")

    assert store.get("odd.mp4").media_origin is None


def test_metadata_that_is_not_an_object_leaves_the_marker_unset(store: SqliteVideoRecordStorage) -> None:
    """`json_extract` over a non-object blob yields NULL rather than raising."""
    _save_with_metadata(store, "array.mp4", "[1, 2, 3]")

    assert store.get("array.mp4").media_origin is None


def test_a_malformed_metadata_blob_does_not_fail_the_query(store: SqliteVideoRecordStorage) -> None:
    """`json_extract` RAISES on unparseable text, and this expression runs on every listed row.

    Unguarded, one malformed blob would fail the whole page — `get_many` deserializes rows in
    a comprehension — rather than the one video. The column is plain TEXT with no CHECK, so
    only convention keeps such a row out; the `json_valid` guard makes that not matter.
    """
    _save_with_metadata(store, "bad.mp4", "not json at all")
    _save_with_metadata(store, "good.mp4", '{"media_origin": "audio_upload"}')

    listed = store.get_many(offset=0, limit=10, order_dir=SQLiteDirection.Descending, user_id="alice", is_admin=False)

    assert {record.video_name: record.media_origin for record in listed.items} == {
        "bad.mp4": None,
        "good.mp4": "audio_upload",
    }


def test_an_overlong_marker_is_dropped_rather_than_echoed_on_every_row(store: SqliteVideoRecordStorage) -> None:
    """Upload metadata is client-supplied and unbounded, and this key now rides every row.

    Without a cap, one upload carrying a huge marker is echoed back on every gallery page
    that includes the video.
    """
    _save_with_metadata(store, "huge.mp4", json.dumps({"media_origin": "A" * 5000}))

    assert store.get("huge.mp4").media_origin is None
    # A marker of a plausible length still passes.
    _save_with_metadata(store, "fine.mp4", json.dumps({"media_origin": "audio_upload"}))
    assert store.get("fine.mp4").media_origin == "audio_upload"
