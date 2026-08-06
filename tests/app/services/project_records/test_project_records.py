"""Tests for the project records service: CRUD, optimistic concurrency, boards, and user isolation."""

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.project_records.project_records_common import (
    ProjectBoardNotFoundError,
    ProjectBoardUnavailableError,
    ProjectRecordConflictError,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
)
from invokeai.app.services.project_records.project_records_sqlite import ProjectRecordsSqlite
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

SYSTEM_USER_ID = "system"


@pytest.fixture
def db() -> SqliteDatabase:
    config = InvokeAIAppConfig(use_memory_db=True)
    return create_mock_sqlite_database(config=config, logger=InvokeAILogger.get_logger())


@pytest.fixture
def project_records(db: SqliteDatabase) -> ProjectRecordsSqlite:
    return ProjectRecordsSqlite(db=db)


@pytest.fixture
def other_user_id(db: SqliteDatabase) -> str:
    users = UserService(db=db)
    user = users.create(
        UserCreateRequest(email="other@example.com", display_name="Other", password="TestPass123", is_admin=False)
    )
    return user.user_id


def test_create_and_get_roundtrip(project_records: ProjectRecordsSqlite) -> None:
    data = {"layout": {"centerViewId": "canvas"}, "widgets": [1, 2, 3], "nested": {"a": None, "b": True}}

    created = project_records.create(SYSTEM_USER_ID, "My Project", data)

    assert created.name == "My Project"
    assert created.revision == 1
    assert created.data == data

    fetched = project_records.get(SYSTEM_USER_ID, created.project_id)
    assert fetched == created


def test_create_with_client_id_and_duplicate_rejected(project_records: ProjectRecordsSqlite) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Imported", {"x": 1}, project_id="project-abc")
    assert created.project_id == "project-abc"

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Imported again", {"x": 2}, project_id="project-abc")


def test_same_project_id_allowed_for_different_users(project_records: ProjectRecordsSqlite, other_user_id: str) -> None:
    project_records.create(SYSTEM_USER_ID, "Mine", {"owner": "system"}, project_id="project-shared-id")
    other = project_records.create(other_user_id, "Theirs", {"owner": "other"}, project_id="project-shared-id")

    assert project_records.get(SYSTEM_USER_ID, "project-shared-id").data == {"owner": "system"}
    assert project_records.get(other_user_id, other.project_id).data == {"owner": "other"}


def test_list_returns_summaries_for_own_projects_only(
    project_records: ProjectRecordsSqlite, other_user_id: str
) -> None:
    first = project_records.create(SYSTEM_USER_ID, "First", {"n": 1})
    second = project_records.create(SYSTEM_USER_ID, "Second", {"n": 2})
    project_records.create(other_user_id, "Not mine", {"n": 3})

    summaries = project_records.list(SYSTEM_USER_ID)

    assert [summary.project_id for summary in summaries] == [first.project_id, second.project_id]
    assert all(not hasattr(summary, "data") for summary in summaries)


def test_update_increments_revision(project_records: ProjectRecordsSqlite) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"v": 1})

    updated = project_records.update(
        SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Renamed", data={"v": 2}
    )

    assert updated.revision == 2
    assert updated.name == "Renamed"
    assert updated.data == {"v": 2}


def test_update_with_stale_revision_raises_conflict(project_records: ProjectRecordsSqlite) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"v": 1})
    project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Project", data={"v": 2})

    with pytest.raises(ProjectRecordConflictError) as exc_info:
        project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Project", data={"v": 3})

    assert exc_info.value.current_revision == 2
    # The conflicting save must not have been applied.
    assert project_records.get(SYSTEM_USER_ID, created.project_id).data == {"v": 2}


def test_update_missing_project_raises_not_found(project_records: ProjectRecordsSqlite) -> None:
    with pytest.raises(ProjectRecordNotFoundError):
        project_records.update(SYSTEM_USER_ID, "does-not-exist", expected_revision=1, name="x", data={})


def test_get_missing_project_raises_not_found(project_records: ProjectRecordsSqlite) -> None:
    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(SYSTEM_USER_ID, "does-not-exist")


def test_delete_is_idempotent(project_records: ProjectRecordsSqlite) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Doomed", {})

    project_records.delete(SYSTEM_USER_ID, created.project_id)
    project_records.delete(SYSTEM_USER_ID, created.project_id)

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(SYSTEM_USER_ID, created.project_id)


def test_users_cannot_touch_each_others_projects(project_records: ProjectRecordsSqlite, other_user_id: str) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Private", {"secret": True})

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(other_user_id, created.project_id)

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.update(other_user_id, created.project_id, expected_revision=1, name="Stolen", data={})

    # Deleting someone else's project is a silent no-op for the other user...
    project_records.delete(other_user_id, created.project_id)
    # ...and the owner's project is untouched.
    assert project_records.get(SYSTEM_USER_ID, created.project_id).data == {"secret": True}


# --- boards -----------------------------------------------------------------------------------


def _insert_board(
    db: SqliteDatabase,
    board_id: str,
    *,
    user_id: str = SYSTEM_USER_ID,
    name: str = "Loose board",
    visibility: str = "private",
) -> str:
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO boards (board_id, board_name, user_id, board_visibility) VALUES (?, ?, ?, ?);",
            (board_id, name, user_id, visibility),
        )
    return board_id


def _board_name(db: SqliteDatabase, board_id: str) -> str | None:
    with db.transaction() as cursor:
        cursor.execute("SELECT board_name FROM boards WHERE board_id = ?;", (board_id,))
        row = cursor.fetchone()
    return None if row is None else row[0]


def _board_count(db: SqliteDatabase) -> int:
    with db.transaction() as cursor:
        cursor.execute("SELECT COUNT(*) FROM boards;")
        return cursor.fetchone()[0]


def _add_image_to_board(db: SqliteDatabase, image_name: str, board_id: str) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO images (image_name, image_origin, image_category, width, height)"
            " VALUES (?, 'internal', 'general', 64, 64);",
            (image_name,),
        )
        cursor.execute("INSERT INTO board_images (board_id, image_name) VALUES (?, ?);", (board_id, image_name))


def test_create_makes_one_board_named_after_the_project(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    before = _board_count(db)

    created = project_records.create(SYSTEM_USER_ID, "My Project", {})

    assert _board_count(db) == before + 1
    assert created.board_id
    assert _board_name(db, created.board_id) == "My Project"
    assert project_records.get_board_id(SYSTEM_USER_ID, created.project_id) == created.board_id


def test_create_claims_a_supplied_board_and_renames_it(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    _insert_board(db, "staging-board", name="Untitled")
    before = _board_count(db)

    created = project_records.create(SYSTEM_USER_ID, "Imported", {}, board_id="staging-board")

    assert created.board_id == "staging-board"
    assert _board_name(db, "staging-board") == "Imported"
    # Claiming reuses the board rather than making another.
    assert _board_count(db) == before


def test_claiming_a_missing_or_foreign_board_reports_it_as_missing(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase, other_user_id: str
) -> None:
    _insert_board(db, "theirs", user_id=other_user_id)

    with pytest.raises(ProjectBoardNotFoundError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="no-such-board")

    # Someone else's board must not be distinguishable from one that does not exist.
    with pytest.raises(ProjectBoardNotFoundError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="theirs")


@pytest.mark.parametrize("visibility", ["shared", "public"])
def test_claiming_a_non_private_board_is_rejected(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase, visibility: str
) -> None:
    _insert_board(db, "visible", visibility=visibility)

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="visible")


def test_claiming_an_explicitly_shared_board_is_rejected(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase, other_user_id: str
) -> None:
    _insert_board(db, "lent-out")
    with db.transaction() as cursor:
        cursor.execute("INSERT INTO shared_boards (board_id, user_id) VALUES ('lent-out', ?);", (other_user_id,))

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="lent-out")


def test_a_board_can_only_ever_be_claimed_once(project_records: ProjectRecordsSqlite, db: SqliteDatabase) -> None:
    _insert_board(db, "contested")
    project_records.create(SYSTEM_USER_ID, "First", {}, board_id="contested")

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(SYSTEM_USER_ID, "Second", {}, board_id="contested")

    assert _board_name(db, "contested") == "First"


def test_a_rejected_create_leaves_no_orphan_board(project_records: ProjectRecordsSqlite, db: SqliteDatabase) -> None:
    project_records.create(SYSTEM_USER_ID, "Taken", {}, project_id="project-taken")
    before = _board_count(db)

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Again", {}, project_id="project-taken")

    # The board insert shares the failed insert's transaction, so it rolled back with it.
    assert _board_count(db) == before


def test_a_rejected_claim_leaves_the_board_name_alone(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    _insert_board(db, "staging", name="Untitled")
    project_records.create(SYSTEM_USER_ID, "Taken", {}, project_id="project-taken")

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Renamer", {}, project_id="project-taken", board_id="staging")

    assert _board_name(db, "staging") == "Untitled"


def test_rename_renames_the_board_but_only_on_a_winning_save(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Before", {})

    project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="After", data={})
    assert _board_name(db, created.board_id) == "After"

    with pytest.raises(ProjectRecordConflictError):
        project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Loser", data={})

    assert _board_name(db, created.board_id) == "After"


def test_delete_removes_the_board_and_its_membership_but_keeps_the_media(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Doomed", {})
    _add_image_to_board(db, "kept.png", created.board_id)

    project_records.delete(SYSTEM_USER_ID, created.project_id)

    assert _board_name(db, created.board_id) is None
    with db.transaction() as cursor:
        cursor.execute("SELECT COUNT(*) FROM board_images WHERE image_name = 'kept.png';")
        assert cursor.fetchone()[0] == 0
        # The image itself survives, uncategorized.
        cursor.execute("SELECT COUNT(*) FROM images WHERE image_name = 'kept.png';")
        assert cursor.fetchone()[0] == 1


def test_summaries_and_records_both_carry_the_board(project_records: ProjectRecordsSqlite) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Listed", {})

    (summary,) = [s for s in project_records.list(SYSTEM_USER_ID) if s.project_id == created.project_id]
    assert summary.board_id == created.board_id


def test_get_board_id_is_user_scoped(project_records: ProjectRecordsSqlite, other_user_id: str) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Private", {})

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get_board_id(other_user_id, created.project_id)


# --- board snapshot ---------------------------------------------------------------------------


def _put_image(
    db: SqliteDatabase,
    name: str,
    board_id: str,
    *,
    category: str = "general",
    starred: bool = False,
    is_intermediate: bool = False,
) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO images (image_name, image_origin, image_category, width, height, starred, is_intermediate)"
            " VALUES (?, 'internal', ?, 64, 64, ?, ?);",
            (name, category, starred, is_intermediate),
        )
        cursor.execute("INSERT INTO board_images (board_id, image_name) VALUES (?, ?);", (board_id, name))


def _put_video(
    db: SqliteDatabase,
    name: str,
    board_id: str,
    *,
    category: str = "general",
    starred: bool = False,
    is_intermediate: bool = False,
) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO videos (video_name, video_origin, video_category, width, height, starred, is_intermediate)"
            " VALUES (?, 'internal', ?, 64, 64, ?, ?);",
            (name, category, starred, is_intermediate),
        )
        cursor.execute("INSERT INTO board_videos (board_id, video_name) VALUES (?, ?);", (board_id, name))


def test_the_snapshot_lists_every_visible_category_of_both_kinds(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Full", {})
    for category in ("general", "control", "mask", "user"):
        _put_image(db, f"{category}.png", created.board_id, category=category)
        _put_video(db, f"{category}.mp4", created.board_id, category=category)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert {(item.kind, item.name, item.category) for item in snapshot.items} == {
        (kind, f"{category}.{ext}", category)
        for category in ("general", "control", "mask", "user")
        for kind, ext in (("image", "png"), ("video", "mp4"))
    }


def test_the_snapshot_excludes_what_the_gallery_does_not_show(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Filtered", {})
    _put_image(db, "shown.png", created.board_id)
    # `other` is the canvas's private category — in neither gallery view, so not board membership.
    _put_image(db, "canvas.png", created.board_id, category="other")
    _put_image(db, "scratch.png", created.board_id, is_intermediate=True)
    _put_video(db, "shown.mp4", created.board_id)
    _put_video(db, "canvas.mp4", created.board_id, category="other")
    _put_video(db, "scratch.mp4", created.board_id, is_intermediate=True)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert [item.name for item in snapshot.items] == ["shown.png", "shown.mp4"]


def test_the_snapshot_excludes_media_on_other_boards(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    mine = project_records.create(SYSTEM_USER_ID, "Mine", {})
    theirs = project_records.create(SYSTEM_USER_ID, "Other", {})
    _put_image(db, "mine.png", mine.board_id)
    _put_image(db, "theirs.png", theirs.board_id)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, mine.project_id)

    assert [item.name for item in snapshot.items] == ["mine.png"]


def test_the_snapshot_carries_starring_and_is_ordered_by_kind_then_name(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Ordered", {})
    _put_image(db, "b.png", created.board_id, starred=True)
    _put_image(db, "a.png", created.board_id)
    _put_video(db, "a.mp4", created.board_id)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert [(item.kind, item.name, item.starred) for item in snapshot.items] == [
        ("image", "a.png", False),
        ("image", "b.png", True),
        ("video", "a.mp4", False),
    ]


def test_a_same_name_image_and_video_are_separate_entries(
    project_records: ProjectRecordsSqlite, db: SqliteDatabase
) -> None:
    """Images and videos are separate namespaces, so one name can legitimately be both."""
    created = project_records.create(SYSTEM_USER_ID, "Twins", {})
    _put_image(db, "twin", created.board_id)
    _put_video(db, "twin", created.board_id)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert [(item.kind, item.name) for item in snapshot.items] == [("image", "twin"), ("video", "twin")]


def test_an_empty_board_snapshots_to_an_empty_list(project_records: ProjectRecordsSqlite) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Empty", {})

    assert project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id).items == []


def test_snapshotting_someone_elses_project_is_not_found(
    project_records: ProjectRecordsSqlite, other_user_id: str
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Private", {})

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get_board_snapshot(other_user_id, created.project_id)
