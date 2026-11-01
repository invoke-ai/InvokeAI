"""Each account's workspace in a project, saved apart from the project content, and the document schema floor that
keeps a client that predates the split from reading or writing half a project."""

from typing import Optional

import pytest
from sqlalchemy import ColumnElement, insert, select

from invokeai.app.services.project_records import project_records_default
from invokeai.app.services.project_records.project_records_common import (
    ProjectCanvasSchemaUnsupportedError,
    ProjectDocumentSchemaDowngradeError,
    ProjectDocumentSchemaUnsupportedError,
    ProjectDocumentTooLargeError,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
    ProjectSplitNeedsWorkspaceError,
    ProjectUpdateWorkspaceRequest,
    ProjectWithWorkspaceDTO,
    ProjectWorkspaceBeforeSplitError,
    ProjectWorkspaceConflictError,
    ProjectWorkspaceDTO,
)
from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import boards
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

    raised = _split(project_records, project_id, {"canvas": {}}, {})

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


def test_a_workspace_save_needs_the_projects_canvas_schema(project_records: ProjectRecordsStorage) -> None:
    """The workspace holds canvas state -- the staging area, the selection -- so the canvas floor applies to it."""
    project_id = project_records.create(
        OWNER,
        "Canvas 3",
        {},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
        minimum_document_schema_version=SPLIT,
        max_document_schema_version=SPLIT,
    ).project_id

    with pytest.raises(ProjectCanvasSchemaUnsupportedError):
        _save(project_records, project_id, None, {"stagingArea": {}})

    saved = project_records.save_workspace(
        OWNER, project_id, None, {"stagingArea": {}}, max_document_schema_version=SPLIT, max_canvas_schema_version=3
    )
    fetched = project_records.get(OWNER, project_id, max_canvas_schema_version=3, max_document_schema_version=SPLIT)
    assert (saved.revision, fetched.workspace) == (1, saved)


def test_a_workspace_has_the_same_size_limit_as_the_content(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    project_records.delete(OWNER, project_id)

    assert _references(database, "project_workspace") == {"kept.png"}


@pytest.mark.sqlite_only
def test_deleting_the_project_drops_its_workspace_references_by_key(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    project_id = _split_project(project_records)
    _save(project_records, project_id, None, {"imageName": "a.png"})

    with capture_statements(database) as captured:
        project_records.delete(OWNER, project_id)

    # By key: not a scan of every workspace reference in the install, under the write lock.
    deletes = [(sql, args) for sql, args in captured if sql.startswith("DELETE FROM media_references")]
    assert deletes
    for sql, args in deletes:
        plan = " ".join(explain_query_plan(database, sql, args))
        assert "owner_kind=? AND user_id=? AND owner_id=?" in plan, plan


# endregion


# region the split: content and workspace in one save


def _split(
    project_records: ProjectRecordsStorage,
    project_id: str,
    content: dict,
    workspace: dict,
    expected_workspace_revision: Optional[int] = None,
) -> ProjectWithWorkspaceDTO:
    """The save that splits a whole project: the content, the raised floor and the member's workspace in one."""
    return project_records.update(
        OWNER,
        project_id,
        expected_revision=1,
        name="Split",
        data=content,
        minimum_document_schema_version=SPLIT,
        max_document_schema_version=SPLIT,
        workspace=ProjectUpdateWorkspaceRequest(data=workspace, expected_revision=expected_workspace_revision),
    )


def test_the_split_stores_the_content_the_floor_and_the_first_workspace_together(
    project_records: ProjectRecordsStorage,
) -> None:
    project_id = project_records.create(OWNER, "Whole", {"whole": True, "layout": "a"}).project_id

    saved = _split(project_records, project_id, {"canvas": {"v": 1}}, {"layout": "a"})

    assert (saved.revision, saved.minimum_document_schema_version, saved.data) == (2, SPLIT, {"canvas": {"v": 1}})
    assert saved.workspace is not None
    assert (saved.workspace.revision, saved.workspace.data) == (1, {"layout": "a"})
    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT) == saved


def test_a_stale_workspace_revision_refuses_the_whole_split(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """Content, floor and references stay as they were: a split never lands by half."""
    project_id = project_records.create(OWNER, "Whole", {"imageName": "whole.png"}).project_id
    # A workspace row beside the whole project, as builds that saved workspaces before the split left them; no
    # save writes one there now.
    with database.begin(write=True) as conn:
        conn.execute(
            insert(project_workspaces).values(project_id=project_id, user_id=OWNER, data='{"imageName":"mine.png"}')
        )
        conn.execute(
            insert(media_references).values(
                owner_kind="project_workspace",
                user_id=OWNER,
                owner_id=project_id,
                media_kind="image",
                media_name="mine.png",
            )
        )

    with pytest.raises(ProjectWorkspaceConflictError) as conflict:
        _split(project_records, project_id, {"imageName": "split.png"}, {"imageName": "lost.png"})

    assert conflict.value.current_revision == 1
    fetched = project_records.get(OWNER, project_id, max_document_schema_version=SPLIT)
    assert (fetched.revision, fetched.minimum_document_schema_version, fetched.data) == (
        1,
        3,
        {"imageName": "whole.png"},
    )
    assert fetched.workspace is not None
    assert (fetched.workspace.revision, fetched.workspace.data) == (1, {"imageName": "mine.png"})
    assert _references(database, "project") == {"whole.png"}
    assert _references(database, "project_workspace") == {"mine.png"}


def test_the_split_indexes_the_media_of_both_halves(project_records: ProjectRecordsStorage, database: Database) -> None:
    project_id = project_records.create(OWNER, "Whole", {"imageName": "whole.png"}).project_id

    _split(project_records, project_id, {"imageName": "layer.png"}, {"imageName": "staged.png"})

    assert _references(database, "project") == {"layer.png"}
    assert _references(database, "project_workspace") == {"staged.png"}


def test_each_half_of_the_split_has_its_own_size_limit(
    project_records: ProjectRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = project_records.create(OWNER, "Whole", {"v": 1}).project_id
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    with pytest.raises(ProjectDocumentTooLargeError):
        _split(project_records, project_id, {"v": 2}, {"layout": "x" * 32})

    fetched = project_records.get(OWNER, project_id)
    assert (fetched.revision, fetched.minimum_document_schema_version, fetched.data, fetched.workspace) == (
        1,
        3,
        {"v": 1},
        None,
    )


def test_the_split_without_the_members_workspace_is_refused_whole(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """Content stored without the member's workspace row would join every later load with a default workspace, so
    no client, whatever it sends, can leave the project that way."""
    project_id = project_records.create(OWNER, "Whole", {"imageName": "whole.png"}).project_id

    with pytest.raises(ProjectSplitNeedsWorkspaceError) as refused:
        project_records.update(
            OWNER,
            project_id,
            expected_revision=1,
            name="Split",
            data={"imageName": "layer.png"},
            minimum_document_schema_version=SPLIT,
            max_document_schema_version=SPLIT,
        )

    assert refused.value.document_floor == 3
    fetched = project_records.get(OWNER, project_id)
    assert (
        fetched.revision,
        fetched.name,
        fetched.minimum_document_schema_version,
        fetched.data,
        fetched.workspace,
    ) == (
        1,
        "Whole",
        3,
        {"imageName": "whole.png"},
        None,
    )
    assert _references(database, "project") == {"whole.png"}

    # Once split, a content save that repeats the floor needs no workspace: the client sends the floor on every save.
    _split(project_records, project_id, {"imageName": "layer.png"}, {"imageName": "staged.png"})
    saved = project_records.update(
        OWNER,
        project_id,
        expected_revision=2,
        name="Split",
        data={"imageName": "edited.png"},
        minimum_document_schema_version=SPLIT,
        max_document_schema_version=SPLIT,
    )
    assert (saved.revision, saved.workspace) == (3, None)
    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace is not None


@pytest.mark.parametrize("requested_floor", [None, 3])
def test_a_workspace_is_refused_beside_a_whole_document(
    project_records: ProjectRecordsStorage, database: Database, requested_floor: Optional[int]
) -> None:
    """A build from before the split could still save the whole project over it, and nothing would read it."""
    project_id = project_records.create(OWNER, "Whole", {"imageName": "whole.png"}).project_id

    with pytest.raises(ProjectWorkspaceBeforeSplitError) as refused:
        project_records.update(
            OWNER,
            project_id,
            expected_revision=1,
            name="Still whole",
            data={"imageName": "edited.png"},
            minimum_document_schema_version=requested_floor,
            max_document_schema_version=SPLIT,
            workspace=ProjectUpdateWorkspaceRequest(data={"imageName": "staged.png"}, expected_revision=None),
        )

    assert refused.value.document_floor == 3
    fetched = project_records.get(OWNER, project_id)
    assert (fetched.revision, fetched.name, fetched.data, fetched.workspace) == (
        1,
        "Whole",
        {"imageName": "whole.png"},
        None,
    )
    assert _references(database, "project") == {"whole.png"}
    assert _references(database, "project_workspace") == set()


def test_a_workspace_alone_is_refused_beside_a_whole_document(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """The workspace save on its own, as the split save: a build from before the split holds the whole project and
    would read none of it."""
    project_id = project_records.create(OWNER, "Whole", {"imageName": "whole.png"}).project_id

    with pytest.raises(ProjectWorkspaceBeforeSplitError) as refused:
        _save(project_records, project_id, None, {"imageName": "staged.png"})

    assert refused.value.document_floor == 3
    assert project_records.get(OWNER, project_id).workspace is None
    assert _references(database, "project_workspace") == set()


def test_a_save_without_a_workspace_carries_none_and_leaves_the_stored_one(
    project_records: ProjectRecordsStorage,
) -> None:
    project_id = _split_project(project_records)
    mine = _save(project_records, project_id, None, {"layout": "a"})

    saved = project_records.update(
        OWNER, project_id, expected_revision=1, name="Project", data={"v": 2}, max_document_schema_version=SPLIT
    )

    assert (saved.revision, saved.workspace) == (2, None)
    assert project_records.get(OWNER, project_id, max_document_schema_version=SPLIT).workspace == mine


# endregion


# region the create: content and the first workspace in one


def _create_with_workspace(
    project_records: ProjectRecordsStorage, content: dict, workspace: dict, **kwargs: str
) -> ProjectWithWorkspaceDTO:
    """A project as a client that writes document schema 4 creates it: with its creator's workspace."""
    return project_records.create(
        OWNER,
        "Created",
        content,
        minimum_document_schema_version=SPLIT,
        max_document_schema_version=SPLIT,
        workspace=workspace,
        **kwargs,
    )


def _rows(database: Database, column: ColumnElement[str]) -> set[str]:
    with database.begin(write=False) as conn:
        return {row[0] for row in conn.execute(select(column)).all()}


def test_a_create_stores_the_content_and_the_first_workspace_together(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """Not a second request that a sign-out between the two could lose."""
    created = _create_with_workspace(project_records, {"imageName": "layer.png"}, {"imageName": "staged.png"})

    assert (created.revision, created.minimum_document_schema_version, created.data) == (
        1,
        SPLIT,
        {"imageName": "layer.png"},
    )
    assert created.workspace is not None
    assert (created.workspace.revision, created.workspace.data) == (1, {"imageName": "staged.png"})
    assert project_records.get(OWNER, created.project_id, max_document_schema_version=SPLIT) == created
    assert _references(database, "project") == {"layer.png"}
    assert _references(database, "project_workspace") == {"staged.png"}


def test_a_create_without_a_workspace_carries_none(project_records: ProjectRecordsStorage) -> None:
    created = project_records.create(OWNER, "Plain", {}, max_document_schema_version=SPLIT)

    assert created.workspace is None
    assert project_records.get(OWNER, created.project_id).workspace is None


def test_a_workspace_is_refused_beside_a_whole_document_on_create(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    """The create's own floor decides, as the save's does: below the split nothing would read the workspace. Refused
    whole: no project, no board, no references."""
    with pytest.raises(ProjectWorkspaceBeforeSplitError) as refused:
        project_records.create(
            OWNER,
            "Whole",
            {"imageName": "whole.png"},
            max_document_schema_version=SPLIT,
            workspace={"imageName": "staged.png"},
        )

    assert refused.value.document_floor == 3
    assert project_records.list(OWNER) == []
    assert _rows(database, boards.c.board_id) == set()
    assert _references(database, "project") == set()
    assert _references(database, "project_workspace") == set()


def test_a_duplicate_id_with_a_workspace_writes_no_workspace_row(
    project_records: ProjectRecordsStorage, database: Database
) -> None:
    project_id = _split_project(project_records, {"imageName": "first.png"})

    with pytest.raises(ProjectRecordExistsError):
        _create_with_workspace(
            project_records, {"imageName": "again.png"}, {"imageName": "lost.png"}, project_id=project_id
        )

    fetched = project_records.get(OWNER, project_id, max_document_schema_version=SPLIT)
    assert (fetched.revision, fetched.data, fetched.workspace) == (1, {"imageName": "first.png"}, None)
    assert _rows(database, project_workspaces.c.project_id) == set()
    assert _references(database, "project") == {"first.png"}
    assert _references(database, "project_workspace") == set()


def test_an_oversized_workspace_refuses_the_whole_create(
    project_records: ProjectRecordsStorage, database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    with pytest.raises(ProjectDocumentTooLargeError):
        _create_with_workspace(project_records, {"v": 1}, {"layout": "x" * 32})

    assert project_records.list(OWNER) == []
    assert _rows(database, boards.c.board_id) == set()
    assert _rows(database, project_workspaces.c.project_id) == set()
    assert _references(database, "project") == set()


def test_a_workspace_write_that_fails_rolls_the_whole_create_back(
    project_records: ProjectRecordsStorage, database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workspace is written in the project's transaction, after its row and references: a failure there leaves
    no project, board or reference behind, which a second transaction after the project's commit could not promise."""

    def failing_write(*args: object, **kwargs: object) -> tuple[int, str]:
        raise RuntimeError("workspace write failed")

    monkeypatch.setattr(project_records_default, "_write_workspace", failing_write)

    with pytest.raises(RuntimeError, match="workspace write failed"):
        _create_with_workspace(project_records, {"imageName": "layer.png"}, {"imageName": "staged.png"})

    assert project_records.list(OWNER) == []
    assert _rows(database, boards.c.board_id) == set()
    assert _rows(database, project_workspaces.c.project_id) == set()
    assert _references(database, "project") == set()
    assert _references(database, "project_workspace") == set()


# endregion
