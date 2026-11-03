"""Video records on every database backend.

The video queries are written apart from the image ones, so they are tested apart: listings and their filters,
per-user isolation, fields, intermediate deletes and the `media_origin` marker.

Per-user isolation covers JPPhoto's code-review finding (PR #9163): when ``board_id`` was omitted from /v1/videos/
and /v1/videos/names, a non-admin caller saw every user's videos.
"""

import json
from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import insert, update
from sqlalchemy.exc import DBAPIError

from invokeai.app.invocations.fields import MetadataFieldValidator
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.pagination import SQLiteDirection
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.video_records.video_records_common import VideoRecordChanges, VideoRecordNotFoundException
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from tests.fixtures.database import capture_statements
from tests.fixtures.races import while_in_flight


@pytest.fixture
def store(database: Database) -> VideoRecordStorage:
    return VideoRecordStorage(database)


def _drop_the_videos_table(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.exec_driver_sql("DROP TABLE videos")


def _save(store: VideoRecordStorage, name: str, user_id: str, is_intermediate: bool = False) -> None:
    store.save(
        video_name=name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        is_intermediate=is_intermediate,
        user_id=user_id,
    )


@pytest.fixture
def seeded_store(store: VideoRecordStorage) -> VideoRecordStorage:
    # Two videos per user; all without board association (the bug occurred when board_id
    # was omitted from the query).
    _save(store, "alice_1.mp4", user_id="alice")
    _save(store, "alice_2.mp4", user_id="alice")
    _save(store, "bob_1.mp4", user_id="bob")
    _save(store, "bob_2.mp4", user_id="bob")
    return store


class TestGetManyOmittedBoardIdMultiuser:
    """get_many() with board_id=None must filter by user_id for non-admin callers."""

    def test_non_admin_only_sees_own_videos(self, seeded_store: VideoRecordStorage) -> None:
        result = seeded_store.get_many(user_id="alice", is_admin=False)
        names = {v.video_name for v in result.items}
        assert names == {"alice_1.mp4", "alice_2.mp4"}
        assert result.total == 2

    def test_admin_sees_every_users_videos(self, seeded_store: VideoRecordStorage) -> None:
        result = seeded_store.get_many(user_id="alice", is_admin=True)
        names = {v.video_name for v in result.items}
        assert names == {"alice_1.mp4", "alice_2.mp4", "bob_1.mp4", "bob_2.mp4"}

    def test_no_user_id_returns_all(self, seeded_store: VideoRecordStorage) -> None:
        # No user_id means the caller is bypassing user filtering entirely (e.g. internal calls).
        result = seeded_store.get_many(user_id=None, is_admin=False)
        names = {v.video_name for v in result.items}
        assert names == {"alice_1.mp4", "alice_2.mp4", "bob_1.mp4", "bob_2.mp4"}


class TestGetVideoNamesOmittedBoardIdMultiuser:
    """get_video_names() with board_id=None must filter by user_id for non-admin callers."""

    def test_non_admin_only_sees_own_videos(self, seeded_store: VideoRecordStorage) -> None:
        result = seeded_store.get_video_names(user_id="alice", is_admin=False)
        assert set(result.video_names) == {"alice_1.mp4", "alice_2.mp4"}
        assert result.total_count == 2

    def test_admin_sees_every_users_videos(self, seeded_store: VideoRecordStorage) -> None:
        result = seeded_store.get_video_names(user_id="alice", is_admin=True)
        assert set(result.video_names) == {"alice_1.mp4", "alice_2.mp4", "bob_1.mp4", "bob_2.mp4"}

    def test_explicit_none_board_still_isolates(self, seeded_store: VideoRecordStorage) -> None:
        # The "none" sentinel (uncategorized) must also apply the user filter — this was the
        # only path that was correct *before* the fix; the test guards against accidental
        # regression there too.
        result = seeded_store.get_video_names(board_id="none", user_id="alice", is_admin=False)
        assert set(result.video_names) == {"alice_1.mp4", "alice_2.mp4"}


class TestDeterministicOrdering:
    @staticmethod
    def _set_same_timestamp(database: Database) -> None:
        with database.begin(write=True) as conn:
            conn.execute(update(videos).values(created_at="2026-07-20 00:00:00.000"))

    def test_same_timestamp_pages_have_stable_mirrored_order(
        self, database: Database, seeded_store: VideoRecordStorage
    ) -> None:
        self._set_same_timestamp(database)

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

    def test_same_timestamp_name_list_has_stable_mirrored_order(
        self, database: Database, seeded_store: VideoRecordStorage
    ) -> None:
        self._set_same_timestamp(database)

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

    def test_board_cover_uses_name_as_same_timestamp_tie_breaker(
        self, database: Database, store: VideoRecordStorage
    ) -> None:
        board = BoardRecordStorage(database).save("Board", "system")
        _save(store, "a.mp4", user_id="system")
        _save(store, "b.mp4", user_id="system")
        self._set_same_timestamp(database)
        for name in ("a.mp4", "b.mp4"):
            BoardVideoRecordStorage(database).add_video_to_board(board.board_id, name)

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

    def test_videos_survive_owner_deletion_and_remain_admin_only(self, database: Database) -> None:
        users = UserService(database)
        store = VideoRecordStorage(database)

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


@pytest.mark.sqlite_only  # A dropped table would stay dropped for every later test on a shared server schema.
def test_get_propagates_a_storage_error_instead_of_reporting_the_row_missing(
    database: Database, store: VideoRecordStorage
):
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
    _drop_the_videos_table(database)

    with pytest.raises(DBAPIError):
        store.get("video-1.mp4")


def test_get_still_reports_a_positively_absent_row_as_missing(store: VideoRecordStorage):
    """The narrowing stays exactly that narrow: absence is still absence."""
    with pytest.raises(VideoRecordNotFoundException):
        store.get("never-existed.mp4")


def test_exists_reports_a_row_get_cannot_deserialize(database: Database, store: VideoRecordStorage):
    """Presence, not readability. `get` would raise on an enum value this version does not know
    — a row written by a newer one — and the refusal path reads that as absence, which would
    report a live video gone."""
    _save(store, "video-1.mp4", "user-1")
    with database.begin(write=True) as conn:
        conn.execute(
            update(videos).where(videos.c.video_name == "video-1.mp4").values(video_category="from_the_future")
        )

    with pytest.raises(ValueError):
        store.get("video-1.mp4")
    assert store.exists("video-1.mp4") is True


@pytest.mark.sqlite_only  # See above.
def test_exists_propagates_a_storage_error(database: Database, store: VideoRecordStorage):
    """ "Could not look" is not "not there" — the caller answers 404 on a False."""
    _drop_the_videos_table(database)

    with pytest.raises(DBAPIError):
        store.exists("video-1.mp4")


@pytest.mark.sqlite_only  # See above.
def test_get_metadata_propagates_a_storage_error(database: Database, store: VideoRecordStorage):
    """As for `get`: a metadata read that fails is not a video that is gone."""
    _save(store, "video-1.mp4", "user-1")
    _drop_the_videos_table(database)

    with pytest.raises(DBAPIError):
        store.get_metadata("video-1.mp4")


def _save_with_metadata(store: VideoRecordStorage, name: str, metadata: str | None) -> None:
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


def test_media_origin_is_projected_out_of_the_metadata_blob(store: VideoRecordStorage) -> None:
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
def test_media_origin_is_none_when_unmarked(store: VideoRecordStorage, metadata: str | None) -> None:
    """`json_extract` over a NULL or key-less blob yields NULL rather than raising."""
    _save_with_metadata(store, "plain.mp4", metadata)

    assert store.get("plain.mp4").media_origin is None


def test_media_origin_survives_a_listing(store: VideoRecordStorage) -> None:
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
def test_a_non_string_marker_does_not_break_the_record(store: VideoRecordStorage, raw: str) -> None:
    """A client may store any JSON value under `media_origin`; the row must still deserialize.

    `MetadataField` validates only that the blob is an object, so `json_extract` can hand
    back an int or a float. Passing one to the `Optional[str]` field raised while building
    the record -- which took out the video's DTO *and* every gallery listing containing it,
    so one odd upload broke the gallery.
    """
    _save_with_metadata(store, "odd.mp4", '{"media_origin": ' + raw + "}")

    assert store.get("odd.mp4").media_origin is None


def test_metadata_that_is_not_an_object_leaves_the_marker_unset(store: VideoRecordStorage) -> None:
    """`json_extract` over a non-object blob yields NULL rather than raising."""
    _save_with_metadata(store, "array.mp4", "[1, 2, 3]")

    assert store.get("array.mp4").media_origin is None


def test_a_malformed_metadata_blob_does_not_fail_the_query(store: VideoRecordStorage) -> None:
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


def test_an_overlong_marker_is_dropped_rather_than_echoed_on_every_row(store: VideoRecordStorage) -> None:
    """Upload metadata is client-supplied and unbounded, and this key now rides every row.

    Without a cap, one upload carrying a huge marker is echoed back on every gallery page
    that includes the video.
    """
    _save_with_metadata(store, "huge.mp4", json.dumps({"media_origin": "A" * 5000}))

    assert store.get("huge.mp4").media_origin is None
    # A marker of a plausible length still passes.
    _save_with_metadata(store, "fine.mp4", json.dumps({"media_origin": "audio_upload"}))
    assert store.get("fine.mp4").media_origin == "audio_upload"


def test_only_videos_still_intermediate_are_deleted_and_reported_in_the_given_order(store: VideoRecordStorage) -> None:
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        _save(store, name, "alice", is_intermediate=True)
    store.update("b.mp4", VideoRecordChanges(is_intermediate=False))

    deleted = store.delete_intermediates_by_names(["c.mp4", "b.mp4", "gone.mp4", "a.mp4"])

    assert deleted == ["c.mp4", "a.mp4"]
    assert [store.exists(name) for name in ("a.mp4", "b.mp4", "c.mp4")] == [False, True, False]


def test_a_promotion_in_flight_is_waited_for_and_keeps_the_record(
    database: Database, store: VideoRecordStorage
) -> None:
    """A delete that meets a promotion still in flight waits for it, and then keeps the promoted record."""
    _save(store, "tmp.mp4", "alice", is_intermediate=True)
    _save(store, "promoted.mp4", "alice", is_intermediate=True)
    deleted: list[str] = []

    def promote(q: Queries) -> None:
        q.videos.update("promoted.mp4", VideoRecordChanges(is_intermediate=False))

    errors = while_in_flight(
        database, promote, lambda: deleted.extend(store.delete_intermediates_by_names(["tmp.mp4", "promoted.mp4"]))
    )

    assert errors == []
    assert deleted == ["tmp.mp4"]
    assert store.get("promoted.mp4").is_intermediate is False


def test_a_guard_narrows_what_is_deleted(store: VideoRecordStorage) -> None:
    _save(store, "a.mp4", "alice", is_intermediate=True)
    _save(store, "b.mp4", "alice", is_intermediate=True)
    asked: list[list[str]] = []

    def guard(q: Queries, names: Sequence[str]) -> list[str]:
        asked.append(list(names))
        return [name for name in names if name != "b.mp4"]

    assert store.delete_intermediates_by_names(["a.mp4", "b.mp4"], guard) == ["a.mp4"]
    assert asked == [["a.mp4", "b.mp4"]]
    assert store.get("b.mp4").is_intermediate is True


class TestListingFilters:
    """Each filter of a listing, on pages and names alike, for an administrator's view."""

    @pytest.fixture
    def board_id(self, database: Database, store: VideoRecordStorage) -> str:
        board = BoardRecordStorage(database).save("Board", "alice")
        # name, category, origin, metadata, starred, intermediate, on the board, created at
        rows = [
            (
                "boarded.mp4",
                ImageCategory.GENERAL,
                ResourceOrigin.INTERNAL,
                '{"prompt": "a 100% Cat_Photo"}',
                True,
                False,
                True,
                "01",
            ),
            (
                "user.mp4",
                ImageCategory.USER,
                ResourceOrigin.INTERNAL,
                '{"prompt": "a 100x catXphoto"}',
                False,
                False,
                False,
                "02",
            ),
            ("external.mp4", ImageCategory.GENERAL, ResourceOrigin.EXTERNAL, None, False, False, False, "03"),
            ("intermediate.mp4", ImageCategory.GENERAL, ResourceOrigin.INTERNAL, None, True, True, True, "04"),
        ]
        for name, category, origin, metadata, starred, intermediate, on_board, day in rows:
            store.save(
                video_name=name,
                video_origin=origin,
                video_category=category,
                width=8,
                height=8,
                duration=1.0,
                fps=8.0,
                has_workflow=False,
                is_intermediate=intermediate,
                starred=starred,
                metadata=metadata,
                user_id="alice",
            )
            with database.begin(write=True) as conn:
                conn.execute(
                    update(videos).where(videos.c.video_name == name).values(created_at=f"2026-01-{day} 00:00:00.000")
                )
            if on_board:
                BoardVideoRecordStorage(database).add_video_to_board(board.board_id, name)
        return board.board_id

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            pytest.param({"board_id": "board"}, {"boarded.mp4", "intermediate.mp4"}, id="one board"),
            pytest.param({"board_id": "none"}, {"user.mp4", "external.mp4"}, id="no board"),
            pytest.param({"categories": [ImageCategory.USER]}, {"user.mp4"}, id="categories"),
            pytest.param({"video_origin": ResourceOrigin.EXTERNAL}, {"external.mp4"}, id="origin"),
            pytest.param({"is_intermediate": False}, {"boarded.mp4", "user.mp4", "external.mp4"}, id="intermediate"),
            # Literal % and _, and either case: the unescaped pattern would also match "100x catXphoto".
            pytest.param({"search_term": "100% cat_"}, {"boarded.mp4"}, id="search"),
        ],
    )
    def test_a_filter_narrows_pages_and_names_alike(
        self, store: VideoRecordStorage, board_id: str, filters: dict[str, Any], expected: set[str]
    ) -> None:
        if filters.get("board_id") == "board":
            filters = {**filters, "board_id": board_id}

        page = store.get_many(limit=10, is_admin=True, **filters)
        names = store.get_video_names(is_admin=True, **filters)

        assert {record.video_name for record in page.items} == expected
        assert page.total == names.total_count == len(expected)
        assert names.starred_count == len(expected & {"boarded.mp4", "intermediate.mp4"})
        assert [record.video_name for record in page.items] == names.video_names

    def test_starred_videos_come_first_and_are_counted(self, store: VideoRecordStorage, board_id: str) -> None:
        names = store.get_video_names(starred_first=True, order_dir=SQLiteDirection.Descending, is_admin=True)
        unstarred_first = store.get_video_names(starred_first=False, order_dir=SQLiteDirection.Ascending, is_admin=True)

        assert names.video_names == ["intermediate.mp4", "boarded.mp4", "external.mp4", "user.mp4"]
        assert names.starred_count == 2
        assert unstarred_first.video_names == ["boarded.mp4", "user.mp4", "external.mp4", "intermediate.mp4"]

    def test_the_cover_is_the_newest_video_that_is_no_intermediate(
        self, store: VideoRecordStorage, board_id: str
    ) -> None:
        cover = store.get_most_recent_video_for_board(board_id)

        assert cover is not None and cover.video_name == "boarded.mp4"


class TestFields:
    def test_save_stores_every_field_and_update_changes_every_field(self, store: VideoRecordStorage) -> None:
        store.save(
            video_name="full.mp4",
            video_origin=ResourceOrigin.EXTERNAL,
            video_category=ImageCategory.CONTROL,
            width=640,
            height=480,
            duration=2.5,
            fps=30.0,
            has_workflow=True,
            is_intermediate=True,
            starred=True,
            session_id="session-1",
            node_id="node-1",
            metadata='{"seed": 1}',
            user_id="user-1",
            video_subfolder="sub/dir",
            project_id="project-1",
        )

        saved = store.get("full.mp4")
        store.update(
            "full.mp4",
            VideoRecordChanges(
                video_category=ImageCategory.USER, session_id="session-2", is_intermediate=False, starred=False
            ),
        )
        updated = store.get("full.mp4")

        assert (
            saved.video_origin,
            saved.video_category,
            saved.width,
            saved.height,
            saved.duration,
            saved.fps,
            saved.has_workflow,
            saved.is_intermediate,
            saved.starred,
            saved.session_id,
            saved.node_id,
            saved.video_subfolder,
            saved.project_id,
        ) == (
            ResourceOrigin.EXTERNAL,
            ImageCategory.CONTROL,
            640,
            480,
            2.5,
            30.0,
            True,
            True,
            True,
            "session-1",
            "node-1",
            "sub/dir",
            "project-1",
        )
        assert store.get_user_id("full.mp4") == "user-1"
        assert store.get_metadata("full.mp4") == MetadataFieldValidator.validate_json('{"seed": 1}')
        assert (updated.video_category, updated.session_id, updated.is_intermediate, updated.starred) == (
            ImageCategory.USER,
            "session-2",
            False,
            False,
        )
        assert updated.node_id == "node-1" and updated.has_workflow is True

    def test_saving_a_taken_name_keeps_the_record_and_returns_its_time(
        self, database: Database, store: VideoRecordStorage
    ) -> None:
        _save(store, "taken.mp4", "alice")
        with database.begin(write=True) as conn:
            conn.execute(
                update(videos).where(videos.c.video_name == "taken.mp4").values(created_at="2026-01-01 00:00:00.000")
            )

        created_at = store.save(
            video_name="taken.mp4",
            video_origin=ResourceOrigin.EXTERNAL,
            video_category=ImageCategory.USER,
            width=1,
            height=1,
            duration=9.0,
            fps=None,
            has_workflow=True,
            user_id="bob",
        )

        record = store.get("taken.mp4")
        assert created_at.isoformat(" ", "milliseconds") == "2026-01-01 00:00:00.000"
        assert (record.video_origin, record.width, store.get_user_id("taken.mp4")) == (
            ResourceOrigin.INTERNAL,
            64,
            "alice",
        )


def test_a_size_backfill_fills_only_sizes_not_known_yet(store: VideoRecordStorage) -> None:
    _save(store, "unknown.mp4", "alice")
    _save(store, "measured.mp4", "alice")
    store.set_file_size_bytes("measured.mp4", 100)

    store.set_file_sizes_bytes({"unknown.mp4": 7, "measured.mp4": 9})

    assert [store.get(name).file_size_bytes for name in ("unknown.mp4", "measured.mp4")] == [7, 100]


@pytest.mark.parametrize("operation", ["delete_intermediates_by_names", "delete_many", "get_subfolders"])
def test_every_name_is_covered_and_no_statement_binds_more_than_sqlite_takes(
    database: Database, store: VideoRecordStorage, operation: str
) -> None:
    """999 is the SQLITE_MAX_VARIABLE_NUMBER default on builds older than 3.32."""
    names = [f"tmp{i:05d}.mp4" for i in range(2 * 500 + 7)]
    with database.begin(write=True) as conn:
        conn.execute(
            insert(videos),
            [
                {
                    "video_name": name,
                    "video_origin": ResourceOrigin.INTERNAL.value,
                    "video_category": ImageCategory.GENERAL.value,
                    "width": 8,
                    "height": 8,
                    "is_intermediate": True,
                    "video_subfolder": "sub",
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
        assert store.get_video_names().video_names == []
    if operation == "delete_intermediates_by_names":
        assert result == names
