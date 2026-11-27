"""Writes of one workflow racing each other.

A write locks the workflow's row before it reads what it decides on, so a write in flight is either seen by the
write that waited for it or finds the row unchanged; see `tests/fixtures/races.py`.
"""

import uuid

import pytest
from sqlalchemy import select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ConflictError
from invokeai.app.services.shared.database.queries.workflows import WorkflowQueries
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowIdConflictError,
    WorkflowMeta,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRevisionConflictError,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage
from tests.fixtures.races import when_called

USER = "user-1"


def _workflow(name: str, image_name: str) -> WorkflowWithoutID:
    return WorkflowWithoutID(
        name=name,
        author="",
        description="",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.User),
        nodes=[{"id": "n1", "data": {"inputs": {"image": {"value": {"image_name": image_name}}}}}],
        edges=[],
    )


def _edit(created: WorkflowRecordDTO, name: str, image_name: str) -> Workflow:
    return Workflow(**_workflow(name, image_name).model_dump(), id=created.workflow_id)


def _references(database: Database, workflow_id: str) -> set[str]:
    with database.begin(write=False) as conn:
        return set(
            conn.execute(
                select(media_references.c.media_name).where(
                    media_references.c.owner_kind == "workflow", media_references.c.owner_id == workflow_id
                )
            ).scalars()
        )


@pytest.mark.parametrize("expects_a_revision", [True, False])
def test_a_save_waits_for_a_save_in_flight(
    workflow_records: WorkflowRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
    expects_a_revision: bool,
) -> None:
    created = workflow_records.create(_workflow("Created", "created.png"), user_id=USER)

    def late_save() -> object:
        revision = 1 if expects_a_revision else None
        edit = _edit(created, "Late", "late.png")
        return workflow_records.update(edit, user_id=USER, expected_revision=revision)

    ended = when_called(monkeypatch, WorkflowQueries, "save", late_save)
    workflow_records.update(_edit(created, "First", "first.png"), user_id=USER, expected_revision=1)
    errors = ended()

    stored = workflow_records.get(created.workflow_id)
    if expects_a_revision:
        (error,) = errors
        assert isinstance(error, WorkflowRevisionConflictError) and error.current_revision == 2
        assert (stored.workflow.name, stored.revision) == ("First", 2)
        assert _references(database, created.workflow_id) == {"first.png"}
    else:
        # A caller that names no revision writes after the save it waited for.
        assert errors == []
        assert (stored.workflow.name, stored.revision) == ("Late", 3)
        assert _references(database, created.workflow_id) == {"late.png"}
    assert lost_races == []


def test_a_visibility_change_rewrites_the_document_of_the_save_it_waited_for(
    workflow_records: WorkflowRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
) -> None:
    created = workflow_records.create(_workflow("Created", "created.png"), user_id=USER)

    ended = when_called(
        monkeypatch,
        WorkflowQueries,
        "save",
        lambda: workflow_records.update_is_public(created.workflow_id, True, user_id=USER),
    )
    workflow_records.update(_edit(created, "Edited", "edited.png"), user_id=USER)

    assert ended() == []
    stored = workflow_records.get(created.workflow_id)
    assert (stored.workflow.name, stored.workflow.tags, stored.is_public, stored.revision) == (
        "Edited",
        "shared",
        True,
        2,
    )
    assert _references(database, created.workflow_id) == {"edited.png"}
    assert lost_races == []


@pytest.mark.parametrize("in_flight", ["save", "delete"])
def test_a_save_and_a_delete_of_one_workflow_wait_for_each_other(
    workflow_records: WorkflowRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
    in_flight: str,
) -> None:
    created = workflow_records.create(_workflow("Created", "created.png"), user_id=USER)

    if in_flight == "save":
        ended = when_called(
            monkeypatch, WorkflowQueries, "save", lambda: workflow_records.delete(created.workflow_id, user_id=USER)
        )
        workflow_records.update(_edit(created, "Edited", "edited.png"), user_id=USER)
        assert ended() == []
    else:
        ended = when_called(
            monkeypatch,
            WorkflowQueries,
            "delete",
            lambda: workflow_records.update(_edit(created, "Edited", "edited.png"), user_id=USER),
        )
        workflow_records.delete(created.workflow_id, user_id=USER)
        assert [type(e) for e in ended()] == [WorkflowNotFoundError]

    with pytest.raises(WorkflowNotFoundError):
        workflow_records.get(created.workflow_id)
    assert _references(database, created.workflow_id) == set()
    assert lost_races == []


@pytest.mark.parametrize("same_content", [True, False])
def test_a_creation_sent_twice_at_once_makes_one_workflow(
    workflow_records: WorkflowRecordsStorage,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    lost_races: list[ConflictError],
    same_content: bool,
) -> None:
    """On a server the second creation looks for the reserved id while the first has not committed, and only its
    insert finds the id taken; it then answers as if it had seen the first creation's record."""
    reserved = str(uuid.uuid4())
    first = _workflow("Created", "first.png")
    second = first if same_content else _workflow("Created", "other.png")
    results: list[WorkflowRecordDTO] = []

    ended = when_called(
        monkeypatch,
        WorkflowQueries,
        "insert",
        lambda: results.append(workflow_records.create(second, user_id=USER, workflow_id=reserved)),
    )
    created = workflow_records.create(first, user_id=USER, workflow_id=reserved)
    errors = ended()

    if same_content:
        assert errors == []
        assert [(result.workflow_id, result.revision) for result in results] == [(reserved, 1)]
    else:
        assert [type(e) for e in errors] == [WorkflowIdConflictError]
    assert workflow_records.get(reserved) == created
    assert _references(database, reserved) == {"first.png"}
    assert lost_races == []
