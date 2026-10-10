"""Each account's workspace in a project, saved apart from the project content, and the document schema floor that
keeps a client that predates the split from reading or writing half a project."""

from typing import Optional

import pytest
from sqlalchemy import insert, select

from invokeai.app.services.project_records.project_records_common import (
    ProjectDocumentSchemaDowngradeError,
    ProjectDocumentSchemaUnsupportedError,
    ProjectDocumentTooLargeError,
    ProjectRecordNotFoundError,
    ProjectWorkspaceConflictError,
    ProjectWorkspaceDTO,
)
from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.database.schema.projects import project_workspaces
from invokeai.app.services.shared.database.schema.users import users
from tests.fixtures.database import capture_statements, explain_query_plan

OWNER = "owner"
SPLIT = 4


@pytest.fixture
def project_records(database: Database) -> ProjectRecordsStorage:
    with database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id=OWNER, email="owner@example.com", password_hash="-"))
    return ProjectRecordsStorage(database)


def _split_project(project_records: ProjectRecordsStorage, content: Optional[dict] = None) -> str:
    """A project as a client that writes document schema 4 creates it."""
    return project_records.create(
        OWNER,
        "Project",
        content or {"canvas": {}},
        minimum_document_schema_version=SPLIT,
        max_document_schema_version=SPLIT,
    ).project_id


def _save(
    project_records: ProjectRecordsStorage, project_id: str, expected: Optional[int], data: dict
) -> ProjectWorkspaceDTO:
    return project_records.save_workspace(OWNER, project_id, expected, data, max_document_schema_version=SPLIT)


def test_content_and_workspace_are_saved_and_versioned_apart(project_records: ProjectRecordsStorage) -> None:
    project_id = _split_project(project_records)

    first = _save(project_records, project_id, None, {"layout": "a"})
    content = project_records.update(
        OWNER,
        project_id,
        expected_revision=1,
        name="Renamed",
        data={"canvas": {"v": 2}},
        max_document_schema_version=SPLIT,
    )
    second = _save(project_records, project_id, first.revision, {"layout": "b"})

    assert (first.revision, second.revision, content.revision) == (1, 2, 2)
    fetched = project_records.get(OWNER, project_id, max_document_schema_version=SPLIT)
    assert (fetched.revision, fetched.data) == (2, {"canvas": {"v": 2}})
    assert fetched.workspace is not None
    assert (fetched.workspace.revision, fetched.workspace.data) == (2, {"layout": "b"})


def test_a_project_without_a_workspace_reads_as_none(project_records: ProjectRecordsStorage) -> None:
    project_id = _split_project(project_records)

    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace is None


@pytest.mark.parametrize(
    ("saved", "expected", "current"),
    [
        ([], 1, None),  # expects a workspace there is none of
        ([{"v": 1}], None, 1),  # a first save racing another tab's first save
        ([{"v": 1}, {"v": 2}], 1, 2),  # a stale save
    ],
)
def test_a_workspace_save_on_another_revision_conflicts_and_writes_nothing(
    project_records: ProjectRecordsStorage, saved: list[dict], expected: Optional[int], current: Optional[int]
) -> None:
    project_id = _split_project(project_records)
    revision = None
    for data in saved:
        revision = _save(project_records, project_id, revision, data).revision

    with pytest.raises(ProjectWorkspaceConflictError) as conflict:
        _save(project_records, project_id, expected, {"v": "lost"})

    assert conflict.value.current_revision == current
    workspace = project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace
    if saved:
        assert workspace is not None
        assert (workspace.revision, workspace.data) == (current, saved[-1])
    else:
        assert workspace is None


def test_a_client_that_predates_the_split_is_refused_a_split_project(project_records: ProjectRecordsStorage) -> None:
    """It would read the content as a whole document and save it back without the workspace."""
    project_id = _split_project(project_records)

    with pytest.raises(ProjectDocumentSchemaUnsupportedError):
        project_records.get(OWNER, project_id)
    with pytest.raises(ProjectDocumentSchemaUnsupportedError):
        project_records.update(OWNER, project_id, expected_revision=1, name="Old", data={"whole": True})
    with pytest.raises(ProjectDocumentSchemaUnsupportedError):
        project_records.save_workspace(OWNER, project_id, None, {}, max_document_schema_version=3)
    fetched = project_records.get(OWNER, project_id, max_document_schema_version=SPLIT)
    assert (fetched.revision, fetched.workspace) == (1, None)


def test_no_save_raises_the_floor_past_what_its_client_understands(project_records: ProjectRecordsStorage) -> None:
    """It would lock the client out of its own project."""
    with pytest.raises(ProjectDocumentSchemaUnsupportedError):
        project_records.create(OWNER, "Ahead", {}, minimum_document_schema_version=5, max_document_schema_version=4)
    assert project_records.list(OWNER) == []

    project_id = _split_project(project_records)
    with pytest.raises(ProjectDocumentSchemaUnsupportedError):
        project_records.update(
            OWNER,
            project_id,
            expected_revision=1,
            name="Ahead",
            data={},
            minimum_document_schema_version=5,
            max_document_schema_version=4,
        )
    assert project_records.list(OWNER)[0].minimum_document_schema_version == SPLIT


def test_a_client_below_the_floor_is_refused_before_its_revision_is_checked(
    project_records: ProjectRecordsStorage,
) -> None:
    """A stale old client must learn it needs updating, not that it has a conflict to resolve."""
    project_id = _split_project(project_records)
    project_records.update(
        OWNER, project_id, expected_revision=1, name="Moved on", data={}, max_document_schema_version=SPLIT
    )

    with pytest.raises(ProjectDocumentSchemaUnsupportedError):
        project_records.update(OWNER, project_id, expected_revision=1, name="Stale", data={"whole": True})


def test_the_first_split_save_raises_the_floor_which_never_falls(project_records: ProjectRecordsStorage) -> None:
    project_id = project_records.create(OWNER, "Schema 3", {"whole": True}).project_id
    assert project_records.list(OWNER)[0].minimum_document_schema_version == 3

    raised = project_records.update(
        OWNER,
        project_id,
        expected_revision=1,
        name="Split",
        data={"canvas": {}},
        minimum_document_schema_version=SPLIT,
        max_document_schema_version=SPLIT,
    )

    assert raised.minimum_document_schema_version == SPLIT
    with pytest.raises(ProjectDocumentSchemaDowngradeError):
        project_records.update(
            OWNER,
            project_id,
            expected_revision=2,
            name="Split",
            data={},
            minimum_document_schema_version=3,
            max_document_schema_version=SPLIT,
        )


def test_only_the_owner_has_a_workspace_in_a_project(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    project_id = _split_project(project_records)
    with database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id="other", email="other@example.com", password_hash="-"))

    with pytest.raises(ProjectRecordNotFoundError):
        project_records.save_workspace("other", project_id, None, {}, max_document_schema_version=SPLIT)


def test_a_workspace_has_the_same_size_limit_as_the_content(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invokeai.app.services.project_records import project_records_default

    project_id = _split_project(project_records)
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    with pytest.raises(ProjectDocumentTooLargeError):
        _save(project_records, project_id, None, {"layout": "x" * 32})
    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace is None


def test_the_listing_says_when_each_half_was_last_saved(project_records: ProjectRecordsStorage) -> None:
    project_id = _split_project(project_records)
    (before,) = project_records.list(OWNER)

    workspace = _save(project_records, project_id, None, {})

    (after,) = project_records.list(OWNER)
    assert before.workspace_updated_at is None
    assert after.workspace_updated_at == workspace.updated_at
    assert after.updated_at == before.updated_at


def test_a_project_is_read_and_listed_with_the_callers_workspace_only(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """Another account's workspace in the project is theirs: neither its data nor when they saved it."""
    project_id = _split_project(project_records)
    with database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id="member", email="member@example.com", password_hash="-"))
        conn.execute(insert(project_workspaces).values(project_id=project_id, user_id="member", data='{"theirs":1}'))

    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace is None
    (listed,) = project_records.list(OWNER)
    assert listed.workspace_updated_at is None

    mine = _save(project_records, project_id, None, {"mine": 1})

    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace == mine
    (listed,) = project_records.list(OWNER)
    assert listed.workspace_updated_at == mine.updated_at


# region media references


def _references(database: Database, owner_kind: str) -> set[str]:
    with database.begin(write=False) as conn:
        rows = conn.execute(
            select(media_references.c.media_name).where(media_references.c.owner_kind == owner_kind)
        ).all()
    return {row[0] for row in rows}


def test_each_half_keeps_the_media_it_names(project_records: ProjectRecordsStorage, database: Database) -> None:
    """A content save must not drop what only the workspace names, such as a staged candidate, or the intermediates
    cleanup would delete it."""
    project_id = _split_project(project_records, {"layer": {"imageName": "layer.png"}})
    _save(project_records, project_id, None, {"staged": {"imageName": "staged.png"}})

    project_records.update(
        OWNER,
        project_id,
        expected_revision=1,
        name="Project",
        data={"layer": {"imageName": "edited.png"}},
        max_document_schema_version=SPLIT,
    )

    assert _references(database, "project") == {"edited.png"}
    assert _references(database, "project_workspace") == {"staged.png"}

    _save(project_records, project_id, 1, {"staged": {"imageName": "restaged.png"}})

    assert _references(database, "project") == {"edited.png"}
    assert _references(database, "project_workspace") == {"restaged.png"}


def test_a_refused_workspace_save_leaves_its_references(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    project_id = _split_project(project_records)
    _save(project_records, project_id, None, {"imageName": "kept.png"})

    with pytest.raises(ProjectWorkspaceConflictError):
        _save(project_records, project_id, None, {"imageName": "lost.png"})

    assert _references(database, "project_workspace") == {"kept.png"}


def test_deleting_the_project_drops_the_workspace_and_its_references(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    project_id = _split_project(project_records)
    _save(project_records, project_id, None, {"imageName": "a.png"})
    kept_id = _split_project(project_records)
    _save(project_records, kept_id, None, {"imageName": "kept.png"})

    with capture_statements(database) as captured:
        project_records.delete(OWNER, project_id)

    assert _references(database, "project_workspace") == {"kept.png"}
    # By key: not a scan of every workspace reference in the install, under the write lock.
    deletes = [(sql, args) for sql, args in captured if sql.startswith("DELETE FROM media_references")]
    assert deletes
    for sql, args in deletes:
        plan = " ".join(explain_query_plan(database, sql, args))
        assert "owner_kind=? AND user_id=? AND owner_id=?" in plan, plan


# endregion
