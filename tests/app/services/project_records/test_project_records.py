"""Project records on every database backend: CRUD, optimistic concurrency, boards, and user isolation."""

from collections.abc import Sequence
from typing import Any, Optional

import pytest
from sqlalchemy import func, insert, select, update

from invokeai.app.services.board_records.board_records_common import (
    BOARD_NAME_MAX_LENGTH,
    BoardChanges,
    BoardRecordProjectOwnedException,
    BoardVisibility,
)
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.project_records import project_records_default
from invokeai.app.services.project_records.project_records_common import (
    ProjectBoardNotFoundError,
    ProjectBoardTooLargeError,
    ProjectBoardUnavailableError,
    ProjectCanvasSchemaDowngradeError,
    ProjectCanvasSchemaUnsupportedError,
    ProjectDocumentInvalidError,
    ProjectDocumentTooLargeError,
    ProjectRecordConflictError,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
)
from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ConflictError
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.boards import BoardQueries
from invokeai.app.services.shared.database.queries.projects import ProjectQueries
from invokeai.app.services.shared.database.schema.boards import board_images, board_videos, boards, shared_boards
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.schema.videos import videos
from tests.fixtures.races import when_called, while_in_flight

SYSTEM_USER_ID = "system"
OTHER_USER_ID = "other"


@pytest.fixture
def project_records(database: Database) -> ProjectRecordsStorage:
    return ProjectRecordsStorage(database)


@pytest.fixture
def other_user_id(database: Database) -> str:
    with database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id=OTHER_USER_ID, email="other@example.com", password_hash="-"))
    return OTHER_USER_ID


def test_create_and_get_roundtrip(project_records: ProjectRecordsStorage) -> None:
    data = {"layout": {"centerViewId": "canvas"}, "widgets": [1, 2, 3], "nested": {"a": None, "b": True}}

    created = project_records.create(SYSTEM_USER_ID, "My Project", data)

    assert created.name == "My Project"
    assert created.revision == 1
    assert created.data == data

    fetched = project_records.get(SYSTEM_USER_ID, created.project_id)
    assert fetched == created


def test_oversized_documents_are_rejected_before_a_board_is_created(
    project_records: ProjectRecordsStorage, database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)
    before = _board_count(database)

    with pytest.raises(ProjectDocumentTooLargeError) as exc_info:
        project_records.create(SYSTEM_USER_ID, "Too large", {"value": "x" * 32})

    assert exc_info.value.max_bytes == 16
    assert _board_count(database) == before


def test_oversized_updates_preserve_the_acknowledged_revision(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"value": "small"})
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    with pytest.raises(ProjectDocumentTooLargeError):
        project_records.update(
            SYSTEM_USER_ID,
            created.project_id,
            expected_revision=created.revision,
            name="Too large",
            data={"value": "x" * 32},
        )

    stored = project_records.get(SYSTEM_USER_ID, created.project_id)
    assert stored.name == "Project"
    assert stored.revision == created.revision
    assert stored.data == {"value": "small"}


def test_document_limit_counts_compact_utf8_bytes(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 14)

    created = project_records.create(SYSTEM_USER_ID, "Exact", {"value": "é"})
    assert created.data == {"value": "é"}

    with pytest.raises(ProjectDocumentTooLargeError) as exc_info:
        project_records.create(SYSTEM_USER_ID, "Over", {"value": "éa"})

    assert exc_info.value.actual_bytes == 15


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "\ud800"])
def test_non_json_safe_document_values_are_rejected(project_records: ProjectRecordsStorage, value: object) -> None:
    with pytest.raises(ProjectDocumentInvalidError):
        project_records.create(SYSTEM_USER_ID, "Invalid", {"value": value})


def test_a_document_is_stored_as_written(project_records: ProjectRecordsStorage) -> None:
    """Text beyond the Basic Multilingual Plane and characters JSON escapes come back unchanged on every backend."""
    data = {"name": 'Kürbis 🎃 \u0000 "quoted" \\ back', "list": ["日本語", 1.5]}

    created = project_records.create(SYSTEM_USER_ID, "Unicode", data)

    assert project_records.get(SYSTEM_USER_ID, created.project_id).data == data


def test_oversized_legacy_documents_remain_readable_and_deletable(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Legacy", {"value": "x" * 32})
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    assert project_records.get(SYSTEM_USER_ID, created.project_id).data == {"value": "x" * 32}
    project_records.delete(SYSTEM_USER_ID, created.project_id)

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(SYSTEM_USER_ID, created.project_id)


def test_create_with_client_id_and_duplicate_rejected(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Imported", {"x": 1}, project_id="project-abc")
    assert created.project_id == "project-abc"

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Imported again", {"x": 2}, project_id="project-abc")


def test_same_project_id_allowed_for_different_users(
    project_records: ProjectRecordsStorage, other_user_id: str
) -> None:
    project_records.create(SYSTEM_USER_ID, "Mine", {"owner": "system"}, project_id="project-shared-id")
    other = project_records.create(other_user_id, "Theirs", {"owner": "other"}, project_id="project-shared-id")

    assert project_records.get(SYSTEM_USER_ID, "project-shared-id").data == {"owner": "system"}
    assert project_records.get(other_user_id, other.project_id).data == {"owner": "other"}


def test_list_returns_summaries_of_own_projects_oldest_first(
    project_records: ProjectRecordsStorage, database: Database, other_user_id: str
) -> None:
    project_records.create(SYSTEM_USER_ID, "Newer", {"n": 1}, project_id="project-a")
    project_records.create(SYSTEM_USER_ID, "Older", {"n": 2}, project_id="project-b")
    project_records.create(other_user_id, "Not mine", {"n": 3})
    with database.begin(write=True) as conn:
        for project_id, created_at in (
            ("project-b", "2026-01-01 00:00:00.000"),
            ("project-a", "2026-01-01 00:00:00.001"),
        ):
            conn.execute(update(projects).where(projects.c.project_id == project_id).values(created_at=created_at))

    summaries = project_records.list(SYSTEM_USER_ID)

    assert [summary.project_id for summary in summaries] == ["project-b", "project-a"]
    assert all(not hasattr(summary, "data") for summary in summaries)


def test_projects_created_in_the_same_millisecond_are_listed_by_id(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """Oldest first; a tie is broken by the project id, as no backend but SQLite keeps an insertion order."""
    for project_id in ("project-b", "project-c", "project-a"):
        project_records.create(SYSTEM_USER_ID, project_id, {}, project_id=project_id)
    with database.begin(write=True) as conn:
        conn.execute(update(projects).values(created_at="2026-01-01 00:00:00.000"))

    assert [summary.project_id for summary in project_records.list(SYSTEM_USER_ID)] == [
        "project-a",
        "project-b",
        "project-c",
    ]


def test_update_increments_revision(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"v": 1})

    updated = project_records.update(
        SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Renamed", data={"v": 2}
    )

    assert updated.revision == 2
    assert updated.name == "Renamed"
    assert updated.data == {"v": 2}
    assert project_records.get(SYSTEM_USER_ID, created.project_id) == updated


def test_a_save_renews_when_the_project_was_updated(project_records: ProjectRecordsStorage, database: Database) -> None:
    # Only the application sets it on a server. (On SQLite a trigger renews it as well.)
    project_records.create(SYSTEM_USER_ID, "Project", {}, project_id="project")
    with database.begin(write=True) as conn:
        conn.execute(update(projects).values(updated_at="2000-01-01 00:00:00.000"))

    saved = project_records.update(SYSTEM_USER_ID, "project", expected_revision=1, name="Project", data={"v": 2})

    assert saved.updated_at > "2000-01-01 00:00:00.000"
    assert project_records.get(SYSTEM_USER_ID, "project").updated_at == saved.updated_at


def test_writes_return_their_row_without_a_post_commit_read(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_post_commit_get(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("writes must not open a second transaction to build their response")

    monkeypatch.setattr(project_records, "get", reject_post_commit_get)

    created = project_records.create(SYSTEM_USER_ID, "Created", {"v": 1})
    updated = project_records.update(
        SYSTEM_USER_ID,
        created.project_id,
        expected_revision=created.revision,
        name="Updated",
        data={"v": 2},
    )

    assert created.revision == 1
    assert created.data == {"v": 1}
    assert updated.revision == 2
    assert updated.data == {"v": 2}


def test_projects_default_to_canvas_schema_v2(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {})

    assert created.minimum_canvas_schema_version == 2
    (summary,) = [s for s in project_records.list(SYSTEM_USER_ID) if s.project_id == created.project_id]
    assert summary.minimum_canvas_schema_version == 2


def test_create_refuses_a_schema_the_client_cannot_edit_without_creating_a_board(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    before = _board_count(database)

    with pytest.raises(ProjectCanvasSchemaUnsupportedError):
        project_records.create(
            SYSTEM_USER_ID,
            "Future",
            {"canvas": {"version": 3}},
            minimum_canvas_schema_version=3,
            max_canvas_schema_version=2,
        )

    assert _board_count(database) == before


def test_an_incapable_client_cannot_read_or_write_a_newer_project(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(
        SYSTEM_USER_ID,
        "Future",
        {"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
    )

    with pytest.raises(ProjectCanvasSchemaUnsupportedError):
        project_records.get(SYSTEM_USER_ID, created.project_id)
    with pytest.raises(ProjectCanvasSchemaUnsupportedError):
        project_records.update(
            SYSTEM_USER_ID,
            created.project_id,
            expected_revision=1,
            name="Overwritten",
            data={"canvas": {"version": 2}},
        )

    preserved = project_records.get(SYSTEM_USER_ID, created.project_id, max_canvas_schema_version=3)
    assert preserved.name == "Future"
    assert preserved.revision == 1
    assert preserved.data == {"canvas": {"version": 3}}


def test_update_atomically_raises_the_canvas_schema_floor_with_the_document(
    project_records: ProjectRecordsStorage,
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"canvas": {"version": 2}})

    updated = project_records.update(
        SYSTEM_USER_ID,
        created.project_id,
        expected_revision=1,
        name="Project v3",
        data={"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
    )

    assert updated.minimum_canvas_schema_version == 3
    assert updated.revision == 2
    assert updated.data == {"canvas": {"version": 3}}
    with pytest.raises(ProjectCanvasSchemaUnsupportedError):
        project_records.get(SYSTEM_USER_ID, created.project_id)


def test_a_stale_save_cannot_raise_the_canvas_schema_floor(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"canvas": {"version": 2}})
    project_records.update(SYSTEM_USER_ID, created.project_id, 1, "Project", {"canvas": {"version": 2}})

    with pytest.raises(ProjectRecordConflictError):
        project_records.update(
            SYSTEM_USER_ID,
            created.project_id,
            expected_revision=1,
            name="Stale v3",
            data={"canvas": {"version": 3}},
            minimum_canvas_schema_version=3,
            max_canvas_schema_version=3,
        )

    preserved = project_records.get(SYSTEM_USER_ID, created.project_id)
    assert preserved.minimum_canvas_schema_version == 2
    assert preserved.revision == 2
    assert preserved.data == {"canvas": {"version": 2}}


def test_a_capable_stale_client_gets_a_revision_conflict_before_floor_downgrade_validation(
    project_records: ProjectRecordsStorage,
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"canvas": {"version": 2}})
    project_records.update(
        SYSTEM_USER_ID,
        created.project_id,
        expected_revision=1,
        name="Raised elsewhere",
        data={"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
    )

    with pytest.raises(ProjectRecordConflictError) as exc_info:
        project_records.update(
            SYSTEM_USER_ID,
            created.project_id,
            expected_revision=1,
            name="Stale local edit",
            data={"canvas": {"version": 2}},
            minimum_canvas_schema_version=2,
            max_canvas_schema_version=3,
        )

    assert exc_info.value.current_revision == 2
    preserved = project_records.get(SYSTEM_USER_ID, created.project_id, max_canvas_schema_version=3)
    assert preserved.minimum_canvas_schema_version == 3
    assert preserved.data == {"canvas": {"version": 3}}


def test_the_canvas_schema_floor_cannot_be_lowered(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(
        SYSTEM_USER_ID,
        "Future",
        {"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
    )

    with pytest.raises(ProjectCanvasSchemaDowngradeError):
        project_records.update(
            SYSTEM_USER_ID,
            created.project_id,
            expected_revision=1,
            name="Downgraded",
            data={"canvas": {"version": 2}},
            minimum_canvas_schema_version=2,
            max_canvas_schema_version=3,
        )

    preserved = project_records.get(SYSTEM_USER_ID, created.project_id, max_canvas_schema_version=3)
    assert preserved.minimum_canvas_schema_version == 3
    assert preserved.revision == 1
    assert preserved.data == {"canvas": {"version": 3}}


def test_update_with_stale_revision_raises_conflict(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Project", {"v": 1})
    project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Project", data={"v": 2})

    with pytest.raises(ProjectRecordConflictError) as exc_info:
        project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Project", data={"v": 3})

    assert exc_info.value.current_revision == 2
    # The conflicting save must not have been applied.
    assert project_records.get(SYSTEM_USER_ID, created.project_id).data == {"v": 2}


def test_update_missing_project_raises_not_found(project_records: ProjectRecordsStorage) -> None:
    with pytest.raises(ProjectRecordNotFoundError):
        project_records.update(SYSTEM_USER_ID, "does-not-exist", expected_revision=1, name="x", data={})


def test_get_missing_project_raises_not_found(project_records: ProjectRecordsStorage) -> None:
    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(SYSTEM_USER_ID, "does-not-exist")


def test_delete_is_idempotent(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Doomed", {})

    project_records.delete(SYSTEM_USER_ID, created.project_id)
    project_records.delete(SYSTEM_USER_ID, created.project_id)

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(SYSTEM_USER_ID, created.project_id)


def test_users_cannot_touch_each_others_projects(project_records: ProjectRecordsStorage, other_user_id: str) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Private", {"secret": True})

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(other_user_id, created.project_id)

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.update(other_user_id, created.project_id, expected_revision=1, name="Stolen", data={})

    # Deleting someone else's project is a silent no-op for the other user...
    project_records.delete(other_user_id, created.project_id)
    # ...and the owner's project is untouched.
    assert project_records.get(SYSTEM_USER_ID, created.project_id).data == {"secret": True}


# region boards


def _insert_board(
    database: Database,
    board_id: str,
    *,
    user_id: str = SYSTEM_USER_ID,
    name: str = "Loose board",
    visibility: str = "private",
    archived: bool = False,
) -> str:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(boards).values(
                board_id=board_id, board_name=name, user_id=user_id, board_visibility=visibility, archived=archived
            )
        )
    return board_id


def _board(database: Database, board_id: str) -> Optional[tuple[str, bool]]:
    """The board's name and archived flag."""
    with database.begin(write=False) as conn:
        row = conn.execute(select(boards.c.board_name, boards.c.archived).where(boards.c.board_id == board_id)).first()
    return None if row is None else (row[0], bool(row[1]))


def _board_count(database: Database) -> int:
    with database.begin(write=False) as conn:
        return conn.execute(select(func.count()).select_from(boards)).scalar_one()


def test_create_makes_one_board_named_after_the_project(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    before = _board_count(database)

    created = project_records.create(SYSTEM_USER_ID, "My Project", {})

    assert _board_count(database) == before + 1
    assert _board(database, created.board_id) == ("My Project", False)
    assert project_records.get_board_id(SYSTEM_USER_ID, created.project_id) == created.board_id


def test_create_claims_a_supplied_board_and_renames_it(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    _insert_board(database, "staging-board", name="Untitled")
    before = _board_count(database)

    created = project_records.create(SYSTEM_USER_ID, "Imported", {}, board_id="staging-board")

    assert created.board_id == "staging-board"
    assert _board(database, "staging-board") == ("Imported", False)
    # Claiming reuses the board rather than making another.
    assert _board_count(database) == before


def test_claiming_a_board_un_archives_it(project_records: ProjectRecordsStorage, database: Database) -> None:
    """A project's board follows the project, and `PATCH /boards/{id}` refuses `archived` on a
    claimed board — so a board claimed while archived would be invisible in every listing with no
    route left to bring it back."""
    _insert_board(database, "staging-board", name="Untitled", archived=True)

    created = project_records.create(SYSTEM_USER_ID, "Imported", {}, board_id="staging-board")

    assert created.board_id == "staging-board"
    assert _board(database, "staging-board") == ("Imported", False)


def test_claiming_a_missing_or_foreign_board_reports_it_as_missing(
    project_records: ProjectRecordsStorage, database: Database, other_user_id: str
) -> None:
    _insert_board(database, "theirs", user_id=other_user_id)

    with pytest.raises(ProjectBoardNotFoundError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="no-such-board")

    # Someone else's board must not be distinguishable from one that does not exist.
    with pytest.raises(ProjectBoardNotFoundError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="theirs")


@pytest.mark.parametrize("visibility", ["shared", "public"])
def test_claiming_a_non_private_board_is_rejected(
    project_records: ProjectRecordsStorage, database: Database, visibility: str
) -> None:
    _insert_board(database, "visible", visibility=visibility)

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="visible")


def test_claiming_an_explicitly_shared_board_is_rejected(
    project_records: ProjectRecordsStorage, database: Database, other_user_id: str
) -> None:
    _insert_board(database, "lent-out")
    with database.begin(write=True) as conn:
        conn.execute(insert(shared_boards).values(board_id="lent-out", user_id=other_user_id))

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(SYSTEM_USER_ID, "P", {}, board_id="lent-out")


def test_a_board_can_only_ever_be_claimed_once(project_records: ProjectRecordsStorage, database: Database) -> None:
    _insert_board(database, "contested")
    project_records.create(SYSTEM_USER_ID, "First", {}, board_id="contested")

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(SYSTEM_USER_ID, "Second", {}, board_id="contested")

    assert _board(database, "contested") == ("First", False)


def test_a_rejected_create_leaves_no_orphan_board(project_records: ProjectRecordsStorage, database: Database) -> None:
    project_records.create(SYSTEM_USER_ID, "Taken", {}, project_id="project-taken")
    before = _board_count(database)

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Again", {}, project_id="project-taken")

    # The board insert shares the failed insert's transaction, so it rolled back with it.
    assert _board_count(database) == before


def test_a_rejected_claim_leaves_the_board_as_it_was(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    _insert_board(database, "staging", name="Untitled", archived=True)
    project_records.create(SYSTEM_USER_ID, "Taken", {}, project_id="project-taken")

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Renamer", {}, project_id="project-taken", board_id="staging")

    assert _board(database, "staging") == ("Untitled", True)


def test_repeating_a_create_says_the_project_exists_rather_than_blaming_the_board(
    project_records: ProjectRecordsStorage, database: Database, other_user_id: str
) -> None:
    """A client whose create response was lost re-sends it to find out whether the first one landed,
    and the answer decides whether it deletes the media it already uploaded. Reporting the board as
    unavailable would read as "somebody else took it" — the opposite of what happened, and the
    reading that gets the uploads thrown away."""
    _insert_board(database, "staging", name="Untitled")
    project_records.create(SYSTEM_USER_ID, "Imported", {}, project_id="project-import", board_id="staging")

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Imported", {}, project_id="project-import", board_id="staging")

    # A different account claiming the same board is still refused: the exclusion is for the project
    # being created, not for board claims in general. (Foreign ownership is caught first, as a 404,
    # so the board is re-pointed at the other account to reach the claimed check.)
    with database.begin(write=True) as conn:
        conn.execute(update(boards).where(boards.c.board_id == "staging").values(user_id=other_user_id))

    with pytest.raises(ProjectBoardUnavailableError):
        project_records.create(other_user_id, "Theirs", {}, project_id="project-import", board_id="staging")


def test_rename_renames_the_board_but_only_on_a_winning_save(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Before", {})

    project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="After", data={})
    assert _board(database, created.board_id) == ("After", False)

    with pytest.raises(ProjectRecordConflictError):
        project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="Loser", data={})

    assert _board(database, created.board_id) == ("After", False)


def test_a_long_project_name_is_cut_to_the_board_name_limit(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    limit = BOARD_NAME_MAX_LENGTH

    created = project_records.create(SYSTEM_USER_ID, "n" * (limit + 1), {})
    assert _board(database, created.board_id) == ("n" * limit, False)

    project_records.update(SYSTEM_USER_ID, created.project_id, expected_revision=1, name="m" * (limit + 1), data={})
    assert _board(database, created.board_id) == ("m" * limit, False)
    assert project_records.get(SYSTEM_USER_ID, created.project_id).name == "m" * (limit + 1)


def test_delete_removes_the_board_and_its_membership_but_keeps_the_media(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Doomed", {})
    _put_image(database, "kept.png", created.board_id)

    project_records.delete(SYSTEM_USER_ID, created.project_id)

    assert _board(database, created.board_id) is None
    with database.begin(write=False) as conn:
        assert conn.execute(select(board_images.c.image_name)).first() is None
        # The image itself survives, uncategorized.
        assert conn.execute(select(images.c.image_name)).scalars().all() == ["kept.png"]


def test_summaries_and_records_both_carry_the_board(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Listed", {})

    (summary,) = [s for s in project_records.list(SYSTEM_USER_ID) if s.project_id == created.project_id]
    assert summary.board_id == created.board_id


def test_get_board_id_is_user_scoped(project_records: ProjectRecordsStorage, other_user_id: str) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Private", {})

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get_board_id(other_user_id, created.project_id)


# endregion

# region writes racing each other
#
# A write locks the rows it decides on before it reads them, a project's before its board's: a write in flight is
# either seen by the write that waited for it or finds what it decided on unchanged, and no two writes deadlock.


def _replace_project(q: Queries, *, minimum_canvas_schema_version: int = 2) -> None:
    """Deletes the project `project` and creates it again under the same id, with the board `new-board`, as another
    client can."""
    project = q.projects.lock(SYSTEM_USER_ID, "project")
    assert project is not None
    q.boards.lock(project.board_id)
    q.projects.delete(SYSTEM_USER_ID, "project")
    q.boards.delete_if_unclaimed(project.board_id)
    q.boards.insert(board_id="new-board", board_name="New", user_id=SYSTEM_USER_ID)
    q.projects.insert(
        project_id="project",
        user_id=SYSTEM_USER_ID,
        name="New",
        data='{"new":true}',
        board_id="new-board",
        minimum_canvas_schema_version=minimum_canvas_schema_version,
    )


@pytest.mark.parametrize("in_flight", ["publish", "delete", "claim"])
def test_a_claim_sees_the_board_change_it_waited_for(
    project_records: ProjectRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
    in_flight: str,
) -> None:
    _insert_board(database, "staging", name="Untitled")

    def claim() -> object:
        return project_records.create(SYSTEM_USER_ID, "Second", {}, project_id="second", board_id="staging")

    refusal: type[Exception]
    if in_flight == "publish":
        ended = when_called(monkeypatch, BoardQueries, "update", claim)
        BoardRecordStorage(database).update("staging", BoardChanges(board_visibility=BoardVisibility.Public))
        errors, refusal = ended(), ProjectBoardUnavailableError
    elif in_flight == "delete":
        # The boards service deletes a board with this one statement.
        errors = while_in_flight(database, lambda q: q.boards.delete_if_unclaimed("staging"), claim)
        refusal = ProjectBoardNotFoundError
    else:
        ended = when_called(monkeypatch, ProjectQueries, "insert", claim)
        project_records.create(SYSTEM_USER_ID, "First", {}, project_id="first", board_id="staging")
        errors, refusal = ended(), ProjectBoardUnavailableError

    assert [type(e) for e in errors] == [refusal]
    assert lost_races == []
    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get(SYSTEM_USER_ID, "second")


@pytest.mark.parametrize("change", ["publish", "delete"])
def test_a_board_change_waits_for_a_claim_in_flight_and_then_finds_the_board_claimed(
    project_records: ProjectRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
    change: str,
) -> None:
    _insert_board(database, "staging", name="Untitled")
    board_records = BoardRecordStorage(database)
    deleted: list[bool] = []

    def change_board() -> None:
        if change == "delete":
            deleted.append(board_records.delete_if_unclaimed("staging"))
        else:
            board_records.update("staging", BoardChanges(board_visibility=BoardVisibility.Public))

    ended = when_called(monkeypatch, ProjectQueries, "insert", change_board)
    project_records.create(SYSTEM_USER_ID, "First", {}, project_id="first", board_id="staging")
    errors = ended()

    if change == "delete":
        assert (errors, deleted) == ([], [False])
    else:
        assert [type(e) for e in errors] == [BoardRecordProjectOwnedException]
    assert lost_races == []
    assert _board(database, "staging") == ("First", False)
    assert project_records.get_board_id(SYSTEM_USER_ID, "first") == "staging"


@pytest.mark.parametrize("in_flight", ["save", "delete"])
def test_a_save_waits_for_a_write_of_its_project_in_flight(
    project_records: ProjectRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
    in_flight: str,
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Created", {"v": 0}, project_id="project")

    def late_save() -> object:
        return project_records.update(SYSTEM_USER_ID, "project", expected_revision=1, name="Late", data={"v": "late"})

    if in_flight == "save":
        ended = when_called(monkeypatch, ProjectQueries, "save", late_save)
        project_records.update(SYSTEM_USER_ID, "project", expected_revision=1, name="First", data={"v": "first"})
        (error,) = ended()
        assert isinstance(error, ProjectRecordConflictError) and error.current_revision == 2
        stored = project_records.get(SYSTEM_USER_ID, "project")
        assert (stored.name, stored.data, stored.revision) == ("First", {"v": "first"}, 2)
        assert _board(database, created.board_id) == ("First", False)
    else:
        ended = when_called(monkeypatch, ProjectQueries, "delete", late_save)
        project_records.delete(SYSTEM_USER_ID, "project")
        assert [type(e) for e in ended()] == [ProjectRecordNotFoundError]
    assert lost_races == []


@pytest.mark.parametrize("client_schema", [2, 3])
def test_a_save_checks_and_writes_the_project_it_waited_for(
    project_records: ProjectRecordsStorage, database: Database, lost_races: list[ConflictError], client_schema: int
) -> None:
    """A save that waited for another client to replace its project under the same id checks the replacement --
    an older client must not write over a project of a newer canvas schema -- and saves into it, board and all."""
    project_records.create(SYSTEM_USER_ID, "Old", {"old": True}, project_id="project")

    errors = while_in_flight(
        database,
        lambda q: _replace_project(q, minimum_canvas_schema_version=3),
        lambda: project_records.update(
            SYSTEM_USER_ID,
            "project",
            expected_revision=1,
            name="Saved",
            data={"saved": True},
            max_canvas_schema_version=client_schema,
        ),
    )

    stored = project_records.get(SYSTEM_USER_ID, "project", max_canvas_schema_version=3)
    if client_schema == 2:
        assert [type(e) for e in errors] == [ProjectCanvasSchemaUnsupportedError]
        assert (stored.data, stored.minimum_canvas_schema_version, stored.revision) == ({"new": True}, 3, 1)
        assert _board(database, "new-board") == ("New", False)
    else:
        assert errors == []
        assert (stored.data, stored.minimum_canvas_schema_version, stored.revision) == ({"saved": True}, 3, 2)
        assert _board(database, "new-board") == ("Saved", False)
    assert lost_races == []


def test_deleting_a_project_deletes_the_board_it_locked(
    project_records: ProjectRecordsStorage, database: Database, lost_races: list[ConflictError]
) -> None:
    """A delete reads the board to delete from the project row it locks: a delete that waited for another client to
    replace the project deletes the replacement's board, not the one it saw first."""
    project_records.create(SYSTEM_USER_ID, "First", {}, project_id="project")

    errors = while_in_flight(database, _replace_project, lambda: project_records.delete(SYSTEM_USER_ID, "project"))

    assert (errors, lost_races) == ([], [])
    assert project_records.list(SYSTEM_USER_ID) == []
    assert _board(database, "new-board") is None


def test_a_resent_claim_waits_for_a_save_of_its_project_in_flight(
    project_records: ProjectRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
) -> None:
    _insert_board(database, "staging", name="Untitled")
    project_records.create(SYSTEM_USER_ID, "Imported", {}, project_id="project", board_id="staging")

    ended = when_called(
        monkeypatch,
        ProjectQueries,
        "save",
        lambda: project_records.create(SYSTEM_USER_ID, "Imported", {}, project_id="project", board_id="staging"),
    )
    project_records.update(SYSTEM_USER_ID, "project", expected_revision=1, name="Saved", data={})

    assert [type(e) for e in ended()] == [ProjectRecordExistsError]
    assert lost_races == []
    assert _board(database, "staging") == ("Saved", False)


def test_a_board_delete_waits_for_a_delete_of_its_project_in_flight(
    project_records: ProjectRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Doomed", {}, project_id="project")
    deleted: list[bool] = []

    ended = when_called(
        monkeypatch,
        ProjectQueries,
        "delete",
        lambda: deleted.append(BoardRecordStorage(database).delete_if_unclaimed(created.board_id)),
    )
    project_records.delete(SYSTEM_USER_ID, "project")

    assert (ended(), deleted, lost_races) == ([], [False], [])
    assert _board(database, created.board_id) is None


# endregion

# region board snapshot


def _put_image(
    database: Database,
    name: str,
    board_id: str,
    *,
    category: str = "general",
    starred: bool = False,
    is_intermediate: bool = False,
) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images).values(
                image_name=name,
                image_origin="internal",
                image_category=category,
                width=64,
                height=64,
                starred=starred,
                is_intermediate=is_intermediate,
            )
        )
        conn.execute(insert(board_images).values(board_id=board_id, image_name=name))


def _put_video(
    database: Database,
    name: str,
    board_id: str,
    *,
    category: str = "general",
    starred: bool = False,
    is_intermediate: bool = False,
) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(videos).values(
                video_name=name,
                video_origin="internal",
                video_category=category,
                width=64,
                height=64,
                starred=starred,
                is_intermediate=is_intermediate,
            )
        )
        conn.execute(insert(board_videos).values(board_id=board_id, video_name=name))


def test_the_snapshot_lists_every_visible_category_of_both_kinds(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Full", {})
    for category in ("general", "control", "mask", "user"):
        _put_image(database, f"{category}.png", created.board_id, category=category)
        _put_video(database, f"{category}.mp4", created.board_id, category=category)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert {(item.kind, item.name, item.category) for item in snapshot.items} == {
        (kind, f"{category}.{ext}", category)
        for category in ("general", "control", "mask", "user")
        for kind, ext in (("image", "png"), ("video", "mp4"))
    }


def test_the_snapshot_excludes_what_the_gallery_does_not_show(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Filtered", {})
    _put_image(database, "shown.png", created.board_id)
    # `other` is the canvas's private category — in neither gallery view, so not board membership.
    _put_image(database, "canvas.png", created.board_id, category="other")
    _put_image(database, "scratch.png", created.board_id, is_intermediate=True)
    _put_video(database, "shown.mp4", created.board_id)
    _put_video(database, "canvas.mp4", created.board_id, category="other")
    _put_video(database, "scratch.mp4", created.board_id, is_intermediate=True)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert [item.name for item in snapshot.items] == ["shown.png", "shown.mp4"]


def test_the_snapshot_excludes_media_on_other_boards(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    mine = project_records.create(SYSTEM_USER_ID, "Mine", {})
    theirs = project_records.create(SYSTEM_USER_ID, "Other", {})
    _put_image(database, "mine.png", mine.board_id)
    _put_image(database, "theirs.png", theirs.board_id)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, mine.project_id)

    assert [item.name for item in snapshot.items] == ["mine.png"]


def test_the_snapshot_carries_starring_and_is_ordered_by_kind_then_name(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Ordered", {})
    # Upper case sorts before lower case: names are compared byte for byte on every backend.
    _put_image(database, "b.png", created.board_id, starred=True)
    _put_image(database, "a.png", created.board_id)
    _put_image(database, "C.png", created.board_id)
    _put_video(database, "a.mp4", created.board_id)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert [(item.kind, item.name, item.starred) for item in snapshot.items] == [
        ("image", "C.png", False),
        ("image", "a.png", False),
        ("image", "b.png", True),
        ("video", "a.mp4", False),
    ]


def test_a_same_name_image_and_video_are_separate_entries(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """Images and videos are separate namespaces, so one name can legitimately be both."""
    created = project_records.create(SYSTEM_USER_ID, "Twins", {})
    _put_image(database, "twin", created.board_id)
    _put_video(database, "twin", created.board_id)

    snapshot = project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)

    assert [(item.kind, item.name) for item in snapshot.items] == [("image", "twin"), ("video", "twin")]


def test_an_empty_board_snapshots_to_an_empty_list(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Empty", {})

    assert project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id).items == []


def test_a_board_over_the_snapshot_limit_is_refused(
    project_records: ProjectRecordsStorage, database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Full", {})
    for name in ("a.png", "b.png"):
        _put_image(database, name, created.board_id)
    _put_video(database, "c.mp4", created.board_id)

    monkeypatch.setattr(project_records_default, "PROJECT_BOARD_SNAPSHOT_MAX_ITEMS", 3)
    assert len(project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id).items) == 3

    monkeypatch.setattr(project_records_default, "PROJECT_BOARD_SNAPSHOT_MAX_ITEMS", 2)
    with pytest.raises(ProjectBoardTooLargeError):
        project_records.get_board_snapshot(SYSTEM_USER_ID, created.project_id)


def test_snapshotting_someone_elses_project_is_not_found(
    project_records: ProjectRecordsStorage, other_user_id: str
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Private", {})

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.get_board_snapshot(other_user_id, created.project_id)


# endregion

# region media references


def _media_references(database: Database, project_id: str) -> set[tuple[str, str]]:
    with database.begin(write=False) as conn:
        rows: Sequence[Sequence[Any]] = conn.execute(
            select(media_references.c.media_kind, media_references.c.media_name).where(
                media_references.c.owner_kind == "project", media_references.c.owner_id == project_id
            )
        ).all()
        return {(kind, name) for kind, name in rows}


def test_saving_a_project_indexes_the_media_it_references(
    database: Database, project_records: ProjectRecordsStorage
) -> None:
    created = project_records.create(
        SYSTEM_USER_ID,
        "Refs",
        {"canvas": {"stagingArea": {"pendingImages": [{"imageName": "staged.png"}]}}, "video": {"video_name": "a.mp4"}},
    )
    assert _media_references(database, created.project_id) == {("image", "staged.png"), ("video", "a.mp4")}

    project_records.update(
        SYSTEM_USER_ID,
        created.project_id,
        expected_revision=1,
        name="Refs",
        data={"layers": [{"image_name": "layer.png"}]},
    )
    assert _media_references(database, created.project_id) == {("image", "layer.png")}


def test_a_refused_save_leaves_the_reference_index_untouched(
    database: Database, project_records: ProjectRecordsStorage
) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Refs", {"image_name": "first.png"})

    with pytest.raises(ProjectRecordConflictError):
        project_records.update(
            SYSTEM_USER_ID, created.project_id, expected_revision=99, name="Refs", data={"image_name": "second.png"}
        )

    assert _media_references(database, created.project_id) == {("image", "first.png")}


def test_a_refused_create_writes_no_references(database: Database, project_records: ProjectRecordsStorage) -> None:
    project_records.create(SYSTEM_USER_ID, "Refs", {"image_name": "first.png"}, project_id="project")

    with pytest.raises(ProjectRecordExistsError):
        project_records.create(SYSTEM_USER_ID, "Refs", {"image_name": "second.png"}, project_id="project")

    assert _media_references(database, "project") == {("image", "first.png")}


def test_deleting_a_missing_project_leaves_references_under_its_id_alone(
    database: Database, project_records: ProjectRecordsStorage
) -> None:
    """References go only with the project they belong to: on a server, a delete that finds no project can run while
    the project is created, and must not take the references that creation writes."""
    with database.begin(write=True) as conn:
        conn.execute(
            insert(media_references).values(
                owner_kind="project", user_id=SYSTEM_USER_ID, owner_id="project", media_kind="image", media_name="a.png"
            )
        )

    project_records.delete(SYSTEM_USER_ID, "project")

    assert _media_references(database, "project") == {("image", "a.png")}


def test_deleting_a_project_drops_its_references(database: Database, project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(SYSTEM_USER_ID, "Refs", {"image_name": "first.png"})

    project_records.delete(SYSTEM_USER_ID, created.project_id)

    assert _media_references(database, created.project_id) == set()


# endregion
