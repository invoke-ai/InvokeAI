"""Library workflows keep the media reference index current, like project documents do."""

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowAccessDeniedError,
    WorkflowCategory,
    WorkflowMeta,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage


@pytest.fixture
def workflow_records(mock_invoker: Invoker) -> SqliteWorkflowRecordsStorage:
    return mock_invoker.services.workflow_records


def _workflow(image_name: str) -> WorkflowWithoutID:
    return WorkflowWithoutID(
        name="Refs",
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


def _references(records: SqliteWorkflowRecordsStorage, workflow_id: str) -> set[tuple[str, str, str]]:
    with records._db.transaction() as cursor:
        cursor.execute(
            "SELECT user_id, media_kind, media_name FROM media_references WHERE owner_kind = 'workflow' AND owner_id = ?;",
            (workflow_id,),
        )
        return {tuple(row) for row in cursor.fetchall()}


def test_create_update_and_delete_keep_the_index_current(workflow_records: SqliteWorkflowRecordsStorage) -> None:
    created = workflow_records.create(_workflow("input.png"), user_id="user-1")
    assert _references(workflow_records, created.workflow_id) == {("user-1", "image", "input.png")}

    workflow_records.update(Workflow(**_workflow("replaced.png").model_dump(), id=created.workflow_id))
    assert _references(workflow_records, created.workflow_id) == {("user-1", "image", "replaced.png")}

    workflow_records.delete(created.workflow_id, user_id="user-1")
    assert _references(workflow_records, created.workflow_id) == set()


def test_an_update_refused_by_ownership_leaves_the_index_alone(workflow_records: SqliteWorkflowRecordsStorage) -> None:
    created = workflow_records.create(_workflow("input.png"), user_id="user-1")

    with pytest.raises(WorkflowAccessDeniedError):
        workflow_records.update(
            Workflow(**_workflow("stolen.png").model_dump(), id=created.workflow_id), user_id="user-2"
        )

    assert _references(workflow_records, created.workflow_id) == {("user-1", "image", "input.png")}


def test_visibility_change_keeps_the_reference_index_of_the_document_it_writes(
    workflow_records: SqliteWorkflowRecordsStorage,
) -> None:
    created = workflow_records.create(_workflow("old.png"), user_id="user-1")

    shared = workflow_records.update_is_public(created.workflow_id, True, user_id="user-1")

    assert shared.is_public is True
    assert _references(workflow_records, created.workflow_id) == {("user-1", "image", "old.png")}
    workflow_records.update(Workflow(**_workflow("new.png").model_dump(), id=created.workflow_id), user_id="user-1")
    assert _references(workflow_records, created.workflow_id) == {("user-1", "image", "new.png")}
