"""Tests for the `last_run_at` column on workflow library records.

`last_run_at` mirrors the existing `opened_at` machinery: a nullable timestamp set only by a
dedicated "touch" method/endpoint, never as a side effect of `get`/`create`/`update`. This lets
the frontend show "Your last run · 2 days ago" on library cards.
"""

from invokeai.app.services.shared.pagination import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowCategory,
    WorkflowMeta,
    WorkflowRecordOrderBy,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage


def create_minimal_user_workflow() -> WorkflowWithoutID:
    """Builds a minimal `WorkflowWithoutID` with `meta.category == "user"`."""
    return WorkflowWithoutID(
        name="Test Workflow",
        author="",
        description="A test workflow",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.User),
        nodes=[],
        edges=[],
    )


def test_update_last_run_at_sets_timestamp(workflow_records: WorkflowRecordsStorage) -> None:
    workflow = create_minimal_user_workflow()
    created = workflow_records.create(workflow=workflow)
    assert created.last_run_at is None

    workflow_records.update_last_run_at(created.workflow_id)

    fetched = workflow_records.get(created.workflow_id)
    assert fetched.last_run_at is not None

    listed = workflow_records.get_many(
        order_by=WorkflowRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Descending,
        categories=None,
        page=0,
        per_page=10,
    )
    row = next(item for item in listed.items if item.workflow_id == created.workflow_id)
    assert row.last_run_at is not None


def test_update_last_run_at_scoped_to_owning_user(workflow_records: WorkflowRecordsStorage) -> None:
    """Mirrors update_opened_at's ownership scoping: passing a mismatched user_id is a no-op,
    and the row updates only once the correct owner's user_id is supplied."""
    workflow = create_minimal_user_workflow()
    created = workflow_records.create(workflow=workflow, user_id="user-a")

    workflow_records.update_last_run_at(created.workflow_id, user_id="user-b")
    assert workflow_records.get(created.workflow_id).last_run_at is None

    workflow_records.update_last_run_at(created.workflow_id, user_id="user-a")
    assert workflow_records.get(created.workflow_id).last_run_at is not None


def test_update_last_run_at_missing_workflow_is_a_no_op(
    workflow_records: WorkflowRecordsStorage,
) -> None:
    """The service method mirrors update_opened_at: it has no existence check and silently does
    nothing for an unknown workflow_id. The router turns a missing workflow into a 404 by calling
    `get()` first (see tests/app/routers/test_multiuser_authorization.py::TestWorkflowMutationAuth
    ::test_update_last_run_at_missing_workflow_404s)."""
    workflow_records.update_last_run_at("does-not-exist")
