"""Library workflows carry a content revision; explicit template updates and retried creations lean on it."""

import uuid

import pytest
from sqlalchemy import insert, select, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.database.schema.workflows import workflow_library
from invokeai.app.services.shared.pagination import SQLiteDirection
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
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage


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


def _references(database: Database, workflow_id: str) -> set[str]:
    with database.begin(write=False) as conn:
        return set(
            conn.execute(
                select(media_references.c.media_name).where(
                    media_references.c.owner_kind == "workflow", media_references.c.owner_id == workflow_id
                )
            ).scalars()
        )


def _bundled_id(workflow_records: WorkflowRecordsStorage) -> str:
    bundled = workflow_records.get_many(
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


def test_every_content_write_advances_the_revision(workflow_records: WorkflowRecordsStorage) -> None:
    created = workflow_records.create(_workflow(), user_id="user-1")
    assert created.revision == 1

    first = workflow_records.update(_with_id(_workflow("Edited"), created.workflow_id), user_id="user-1")
    assert first.revision == 2

    # Legacy callers that carry no expected revision still write, and still advance it.
    second = workflow_records.update(_with_id(_workflow("Edited again"), created.workflow_id))
    assert second.revision == 3
    assert workflow_records.get(created.workflow_id).workflow.name == "Edited again"

    listed = workflow_records.get_many(
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


def test_a_stale_expected_revision_is_refused_and_writes_nothing(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    created = workflow_records.create(_workflow(image_name="v1.png"), user_id="user-1")
    workflow_records.update(_with_id(_workflow(image_name="v2.png"), created.workflow_id), user_id="user-1")

    with pytest.raises(WorkflowRevisionConflictError) as conflict:
        workflow_records.update(
            _with_id(_workflow(image_name="stale.png"), created.workflow_id),
            user_id="user-1",
            expected_revision=1,
        )

    assert conflict.value.current_revision == 2
    assert conflict.value.expected_revision == 1
    current = workflow_records.get(created.workflow_id)
    assert current.revision == 2
    assert _references(database, created.workflow_id) == {"v2.png"}

    accepted = workflow_records.update(
        _with_id(_workflow(image_name="v3.png"), created.workflow_id), user_id="user-1", expected_revision=2
    )
    assert accepted.revision == 3
    assert _references(database, created.workflow_id) == {"v3.png"}


def test_bookkeeping_writes_leave_the_revision_alone(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    created = workflow_records.create(_workflow(image_name="in.png"), user_id="user-1")

    workflow_records.update_opened_at(created.workflow_id, user_id="user-1")
    workflow_records.update_last_run_at(created.workflow_id, user_id="user-1")
    shared = workflow_records.update_is_public(created.workflow_id, True, user_id="user-1")

    assert shared.revision == 1
    assert shared.is_public is True
    assert "shared" in shared.workflow.tags
    assert _references(database, created.workflow_id) == {"in.png"}
    # An editor holding revision 1 is therefore not asked to resolve a conflict it cannot see.
    assert workflow_records.update(_with_id(_workflow(), created.workflow_id), expected_revision=1).revision == 2


def test_ownership_is_enforced_atomically_with_the_write(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    created = workflow_records.create(_workflow(image_name="mine.png"), user_id="user-1")

    with pytest.raises(WorkflowAccessDeniedError):
        workflow_records.update(_with_id(_workflow(image_name="stolen.png"), created.workflow_id), user_id="user-2")
    with pytest.raises(WorkflowAccessDeniedError):
        workflow_records.delete(created.workflow_id, user_id="user-2")

    current = workflow_records.get(created.workflow_id)
    assert current.revision == 1
    assert _references(database, created.workflow_id) == {"mine.png"}

    with pytest.raises(WorkflowNotFoundError):
        workflow_records.update(_with_id(_workflow(), "missing"), user_id="user-1")


def test_bundled_workflows_are_immutable_whatever_the_request_claims(workflow_records: WorkflowRecordsStorage) -> None:
    bundled_id = _bundled_id(workflow_records)
    before = workflow_records.get(bundled_id)

    # The body says "user"; the stored record decides.
    with pytest.raises(WorkflowImmutableError):
        workflow_records.update(_with_id(_workflow("Hijacked"), bundled_id))
    with pytest.raises(WorkflowImmutableError):
        workflow_records.update(
            _with_id(_workflow("Hijacked"), bundled_id), user_id=before.user_id, expected_revision=1
        )
    with pytest.raises(WorkflowImmutableError):
        workflow_records.delete(bundled_id)

    after = workflow_records.get(bundled_id)
    assert after.workflow == before.workflow
    assert after.revision == before.revision


def test_bundled_sync_advances_the_revision_only_when_content_changes(
    workflow_records: WorkflowRecordsStorage,
    database: Database,
) -> None:
    bundled_id = _bundled_id(workflow_records)
    assert workflow_records.get(bundled_id).revision == 1

    # A second start with unchanged bundles is a no-op for every record.
    workflow_records._sync_default_workflows()
    assert workflow_records.get(bundled_id).revision == 1

    stale = workflow_records.get(bundled_id).workflow.model_copy(update={"notes": "stale copy"})
    with database.begin(write=True) as conn:
        conn.execute(
            update(workflow_library)
            .where(workflow_library.c.workflow_id == bundled_id)
            .values(workflow=stale.model_dump_json())
        )
    workflow_records._sync_default_workflows()
    resynced = workflow_records.get(bundled_id)
    assert resynced.revision == 2
    assert resynced.workflow.notes != "stale copy"


def test_a_reserved_id_makes_creation_retry_safe(workflow_records: WorkflowRecordsStorage, database: Database) -> None:
    reserved = str(uuid.uuid4())
    workflow = _workflow(image_name="first.png")

    created = workflow_records.create(workflow, user_id="user-1", workflow_id=reserved)
    retried = workflow_records.create(workflow, user_id="user-1", workflow_id=reserved)

    assert created.workflow_id == reserved
    assert retried.workflow_id == reserved
    assert retried.revision == 1
    assert _references(database, reserved) == {"first.png"}
    listed = workflow_records.get_many(
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


def test_a_reserved_id_that_names_something_else_is_a_conflict(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    reserved = str(uuid.uuid4())
    workflow_records.create(_workflow(image_name="first.png"), user_id="user-1", workflow_id=reserved)

    with pytest.raises(WorkflowIdConflictError):
        workflow_records.create(_workflow(image_name="other.png"), user_id="user-1", workflow_id=reserved)
    with pytest.raises(WorkflowIdConflictError):
        workflow_records.create(_workflow(image_name="first.png"), user_id="user-2", workflow_id=reserved)
    with pytest.raises(ValueError):
        workflow_records.create(_workflow(), user_id="user-1", workflow_id="not-a-uuid")

    assert workflow_records.get(reserved).user_id == "user-1"
    assert _references(database, reserved) == {"first.png"}


def test_creation_without_a_reserved_id_keeps_generating_ids(workflow_records: WorkflowRecordsStorage) -> None:
    first = workflow_records.create(_workflow(), user_id="user-1")
    second = workflow_records.create(_workflow(), user_id="user-1")

    assert first.workflow_id != second.workflow_id
    assert uuid.UUID(first.workflow_id)


def test_unsharing_removes_the_shared_tag_sharing_added(workflow_records: WorkflowRecordsStorage) -> None:
    created = workflow_records.create(_workflow().model_copy(update={"tags": "mine"}), user_id="user-1")

    workflow_records.update_is_public(created.workflow_id, True, user_id="user-1")
    unshared = workflow_records.update_is_public(created.workflow_id, False, user_id="user-1")

    assert (unshared.is_public, unshared.workflow.tags) == (False, "mine")
    assert workflow_records.get(created.workflow_id) == unshared


def test_a_visibility_change_leaves_others_and_bundled_workflows_alone(
    workflow_records: WorkflowRecordsStorage,
) -> None:
    created = workflow_records.create(_workflow(), user_id="user-1")
    bundled_id = _bundled_id(workflow_records)

    refused = workflow_records.update_is_public(created.workflow_id, True, user_id="user-2")
    bundled = workflow_records.update_is_public(bundled_id, True)

    assert (refused.is_public, workflow_records.get(created.workflow_id).is_public) == (False, False)
    assert (bundled.is_public, workflow_records.get(bundled_id).is_public) == (False, False)
    with pytest.raises(WorkflowNotFoundError):
        workflow_records.update_is_public("missing", True)


def test_a_start_removes_bundled_workflows_no_longer_shipped(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    retired = workflow_records.get(_bundled_id(workflow_records)).workflow.model_copy(update={"id": "default_retired"})
    with database.begin(write=True) as conn:
        conn.execute(insert(workflow_library).values(workflow_id=retired.id, workflow=retired.model_dump_json()))

    workflow_records._sync_default_workflows()

    with pytest.raises(WorkflowNotFoundError):
        workflow_records.get(retired.id)


def test_writes_into_the_default_category_are_refused(workflow_records: WorkflowRecordsStorage) -> None:
    created = workflow_records.create(_workflow(), user_id="user-1")
    bundled_kind = _workflow().model_copy(
        update={"meta": WorkflowMeta(version="3.0.0", category=WorkflowCategory.Default)}
    )

    with pytest.raises(ValueError):
        workflow_records.create(bundled_kind, user_id="user-1")
    with pytest.raises(ValueError):
        workflow_records.update(_with_id(bundled_kind, created.workflow_id), user_id="user-1")


def test_a_workflow_without_an_owner_belongs_to_the_system_account(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    created = workflow_records.create(_workflow(image_name="in.png"), user_id="system")
    with database.begin(write=True) as conn:
        conn.execute(
            update(workflow_library).where(workflow_library.c.workflow_id == created.workflow_id).values(user_id=None)
        )

    with pytest.raises(WorkflowAccessDeniedError):
        workflow_records.delete(created.workflow_id, user_id="user-1")
    workflow_records.delete(created.workflow_id, user_id="system")

    assert _references(database, created.workflow_id) == set()
