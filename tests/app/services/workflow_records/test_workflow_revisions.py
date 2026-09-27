"""Library workflows carry a content revision; explicit template updates and retried creations lean on it."""

import uuid

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowAccessDeniedError,
    WorkflowCategory,
    WorkflowIdConflictError,
    WorkflowImmutableError,
    WorkflowMeta,
    WorkflowNotFoundError,
    WorkflowRecordOrderBy,
    WorkflowRevisionConflictError,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage


@pytest.fixture
def records(mock_invoker: Invoker) -> SqliteWorkflowRecordsStorage:
    return mock_invoker.services.workflow_records


def _workflow(name: str = "Template", image_name: str | None = None) -> WorkflowWithoutID:
    inputs = {"image": {"value": {"image_name": image_name}}} if image_name else {}
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
        nodes=[{"id": "n1", "data": {"inputs": inputs}}],
        edges=[],
    )


def _with_id(workflow: WorkflowWithoutID, workflow_id: str) -> Workflow:
    return Workflow(**workflow.model_dump(), id=workflow_id)


def _references(records: SqliteWorkflowRecordsStorage, workflow_id: str) -> set[str]:
    with records._db.transaction() as cursor:
        cursor.execute(
            "SELECT media_name FROM media_references WHERE owner_kind = 'workflow' AND owner_id = ?;",
            (workflow_id,),
        )
        return {row[0] for row in cursor.fetchall()}


def _bundled_id(records: SqliteWorkflowRecordsStorage) -> str:
    bundled = records.get_many(
        order_by=WorkflowRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
        categories=[WorkflowCategory.Default],
        page=0,
        per_page=1,
        query=None,
        tags=None,
        has_been_opened=None,
    ).items
    assert bundled, "the default workflows are synced on start"
    return bundled[0].workflow_id


def test_every_content_write_advances_the_revision(records: SqliteWorkflowRecordsStorage) -> None:
    created = records.create(_workflow(), user_id="user-1")
    assert created.revision == 1

    first = records.update(_with_id(_workflow("Edited"), created.workflow_id), user_id="user-1")
    assert first.revision == 2

    # Legacy callers that carry no expected revision still write, and still advance it.
    second = records.update(_with_id(_workflow("Edited again"), created.workflow_id))
    assert second.revision == 3
    assert records.get(created.workflow_id).workflow.name == "Edited again"

    listed = records.get_many(
        order_by=WorkflowRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Descending,
        categories=[WorkflowCategory.User],
        page=0,
        per_page=10,
        query=None,
        tags=None,
        has_been_opened=None,
    ).items
    assert next(item.revision for item in listed if item.workflow_id == created.workflow_id) == 3


def test_a_stale_expected_revision_is_refused_and_writes_nothing(records: SqliteWorkflowRecordsStorage) -> None:
    created = records.create(_workflow(image_name="v1.png"), user_id="user-1")
    records.update(_with_id(_workflow(image_name="v2.png"), created.workflow_id), user_id="user-1")

    with pytest.raises(WorkflowRevisionConflictError) as conflict:
        records.update(
            _with_id(_workflow(image_name="stale.png"), created.workflow_id),
            user_id="user-1",
            expected_revision=1,
        )

    assert conflict.value.current_revision == 2
    assert conflict.value.expected_revision == 1
    current = records.get(created.workflow_id)
    assert current.revision == 2
    assert _references(records, created.workflow_id) == {"v2.png"}

    accepted = records.update(
        _with_id(_workflow(image_name="v3.png"), created.workflow_id), user_id="user-1", expected_revision=2
    )
    assert accepted.revision == 3
    assert _references(records, created.workflow_id) == {"v3.png"}


def test_bookkeeping_writes_leave_the_revision_alone(records: SqliteWorkflowRecordsStorage) -> None:
    created = records.create(_workflow(image_name="in.png"), user_id="user-1")

    records.update_opened_at(created.workflow_id, user_id="user-1")
    records.update_last_run_at(created.workflow_id, user_id="user-1")
    shared = records.update_is_public(created.workflow_id, True, user_id="user-1")

    assert shared.revision == 1
    assert shared.is_public is True
    assert "shared" in shared.workflow.tags
    assert _references(records, created.workflow_id) == {"in.png"}
    # An editor holding revision 1 is therefore not asked to resolve a conflict it cannot see.
    assert records.update(_with_id(_workflow(), created.workflow_id), expected_revision=1).revision == 2


def test_ownership_is_enforced_atomically_with_the_write(records: SqliteWorkflowRecordsStorage) -> None:
    created = records.create(_workflow(image_name="mine.png"), user_id="user-1")

    with pytest.raises(WorkflowAccessDeniedError):
        records.update(_with_id(_workflow(image_name="stolen.png"), created.workflow_id), user_id="user-2")
    with pytest.raises(WorkflowAccessDeniedError):
        records.delete(created.workflow_id, user_id="user-2")

    current = records.get(created.workflow_id)
    assert current.revision == 1
    assert _references(records, created.workflow_id) == {"mine.png"}

    with pytest.raises(WorkflowNotFoundError):
        records.update(_with_id(_workflow(), "missing"), user_id="user-1")


def test_bundled_workflows_are_immutable_whatever_the_request_claims(records: SqliteWorkflowRecordsStorage) -> None:
    bundled_id = _bundled_id(records)
    before = records.get(bundled_id)

    # The body says "user"; the stored record decides.
    with pytest.raises(WorkflowImmutableError):
        records.update(_with_id(_workflow("Hijacked"), bundled_id))
    with pytest.raises(WorkflowImmutableError):
        records.update(_with_id(_workflow("Hijacked"), bundled_id), user_id=before.user_id, expected_revision=1)
    with pytest.raises(WorkflowImmutableError):
        records.delete(bundled_id)

    after = records.get(bundled_id)
    assert after.workflow == before.workflow
    assert after.revision == before.revision


def test_bundled_sync_advances_the_revision_only_when_content_changes(
    records: SqliteWorkflowRecordsStorage,
) -> None:
    bundled_id = _bundled_id(records)
    assert records.get(bundled_id).revision == 1

    # A second start with unchanged bundles is a no-op for every record.
    records._sync_default_workflows()
    assert records.get(bundled_id).revision == 1

    with records._db.transaction() as cursor:
        cursor.execute(
            "UPDATE workflow_library SET workflow = json_set(workflow, '$.notes', 'stale copy') WHERE workflow_id = ?;",
            (bundled_id,),
        )
    records._sync_default_workflows()
    resynced = records.get(bundled_id)
    assert resynced.revision == 2
    assert resynced.workflow.notes != "stale copy"


def test_a_reserved_id_makes_creation_retry_safe(records: SqliteWorkflowRecordsStorage) -> None:
    reserved = str(uuid.uuid4())
    workflow = _workflow(image_name="first.png")

    created = records.create(workflow, user_id="user-1", workflow_id=reserved)
    retried = records.create(workflow, user_id="user-1", workflow_id=reserved)

    assert created.workflow_id == reserved
    assert retried.workflow_id == reserved
    assert retried.revision == 1
    assert _references(records, reserved) == {"first.png"}
    listed = records.get_many(
        order_by=WorkflowRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Descending,
        categories=[WorkflowCategory.User],
        page=0,
        per_page=None,
        query=None,
        tags=None,
        has_been_opened=None,
    )
    assert [item.workflow_id for item in listed.items].count(reserved) == 1


def test_a_reserved_id_that_names_something_else_is_a_conflict(records: SqliteWorkflowRecordsStorage) -> None:
    reserved = str(uuid.uuid4())
    records.create(_workflow(image_name="first.png"), user_id="user-1", workflow_id=reserved)

    with pytest.raises(WorkflowIdConflictError):
        records.create(_workflow(image_name="other.png"), user_id="user-1", workflow_id=reserved)
    with pytest.raises(WorkflowIdConflictError):
        records.create(_workflow(image_name="first.png"), user_id="user-2", workflow_id=reserved)
    with pytest.raises(ValueError):
        records.create(_workflow(), user_id="user-1", workflow_id="not-a-uuid")

    assert records.get(reserved).user_id == "user-1"
    assert _references(records, reserved) == {"first.png"}


def test_creation_without_a_reserved_id_keeps_generating_ids(records: SqliteWorkflowRecordsStorage) -> None:
    first = records.create(_workflow(), user_id="user-1")
    second = records.create(_workflow(), user_id="user-1")

    assert first.workflow_id != second.workflow_id
    assert uuid.UUID(first.workflow_id)
