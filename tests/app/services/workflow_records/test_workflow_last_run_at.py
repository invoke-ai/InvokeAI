"""Tests for the `last_run_at` column on workflow library records.

`last_run_at` mirrors the existing `opened_at` machinery: a nullable timestamp set only by a
dedicated "touch" method/endpoint, never as a side effect of `get`/`create`/`update`. This lets
the frontend show "Your last run · 2 days ago" on library cards.
"""

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowCategory,
    WorkflowMeta,
    WorkflowRecordOrderBy,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage


@pytest.fixture
def workflow_records_service(mock_invoker: Invoker) -> SqliteWorkflowRecordsStorage:
    return mock_invoker.services.workflow_records


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


def test_update_last_run_at_sets_timestamp(workflow_records_service: SqliteWorkflowRecordsStorage) -> None:
    workflow = create_minimal_user_workflow()
    created = workflow_records_service.create(workflow=workflow)
    assert created.last_run_at is None

    workflow_records_service.update_last_run_at(created.workflow_id)

    fetched = workflow_records_service.get(created.workflow_id)
    assert fetched.last_run_at is not None

    listed = workflow_records_service.get_many(
        order_by=WorkflowRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Descending,
        categories=None,
        page=0,
        per_page=10,
    )
    row = next(item for item in listed.items if item.workflow_id == created.workflow_id)
    assert row.last_run_at is not None
