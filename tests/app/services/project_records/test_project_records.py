"""Tests for the project records service: CRUD, optimistic concurrency, and user isolation."""

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.project_records.project_records_common import (
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
