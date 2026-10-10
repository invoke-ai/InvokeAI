"""Library workflows keep the media reference index current, like project documents do."""

from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowAccessDeniedError,
    WorkflowCategory,
    WorkflowMeta,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage


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


def _references(database: Database, workflow_id: str) -> set[tuple[str, str, str]]:
    with database.begin(write=False) as conn:
        rows: Sequence[Sequence[Any]] = conn.execute(
            select(media_references.c.user_id, media_references.c.media_kind, media_references.c.media_name).where(
                media_references.c.owner_kind == "workflow", media_references.c.owner_id == workflow_id
            )
        ).all()
    return {(user_id, kind, name) for user_id, kind, name in rows}


def test_create_update_and_delete_keep_the_index_current(
    database: Database, workflow_records: WorkflowRecordsStorage
) -> None:
    created = workflow_records.create(_workflow("input.png"), user_id="user-1")
    assert _references(database, created.workflow_id) == {("user-1", "image", "input.png")}

    workflow_records.update(Workflow(**_workflow("replaced.png").model_dump(), id=created.workflow_id))
    assert _references(database, created.workflow_id) == {("user-1", "image", "replaced.png")}

    workflow_records.delete(created.workflow_id, user_id="user-1")
    assert _references(database, created.workflow_id) == set()


def test_an_update_refused_by_ownership_leaves_the_index_alone(
    database: Database, workflow_records: WorkflowRecordsStorage
) -> None:
    created = workflow_records.create(_workflow("input.png"), user_id="user-1")

    with pytest.raises(WorkflowAccessDeniedError):
        workflow_records.update(
            Workflow(**_workflow("stolen.png").model_dump(), id=created.workflow_id), user_id="user-2"
        )

    assert _references(database, created.workflow_id) == {("user-1", "image", "input.png")}


def test_visibility_change_keeps_the_reference_index_of_the_document_it_writes(
    database: Database, workflow_records: WorkflowRecordsStorage
) -> None:
    created = workflow_records.create(_workflow("old.png"), user_id="user-1")

    shared = workflow_records.update_is_public(created.workflow_id, True, user_id="user-1")

    assert shared.is_public is True
    assert _references(database, created.workflow_id) == {("user-1", "image", "old.png")}
    workflow_records.update(Workflow(**_workflow("new.png").model_dump(), id=created.workflow_id), user_id="user-1")
    assert _references(database, created.workflow_id) == {("user-1", "image", "new.png")}


def test_a_delete_naming_no_account_drops_the_owners_references(
    database: Database, workflow_records: WorkflowRecordsStorage
) -> None:
    created = workflow_records.create(_workflow("input.png"), user_id="user-1")

    workflow_records.delete(created.workflow_id)  # as an administrator deletes it

    assert _references(database, created.workflow_id) == set()
