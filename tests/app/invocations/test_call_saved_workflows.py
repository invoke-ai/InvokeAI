from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from invokeai.app.services.users.users_common import UserDTO
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowMeta,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
)


def build_workflow_record_dto(
    *,
    workflow_id: str = "workflow-123",
    user_id: str = "owner-1",
    category: WorkflowCategory = WorkflowCategory.User,
    is_public: bool = False,
) -> WorkflowRecordDTO:
    workflow = Workflow(
        id=workflow_id,
        name="Saved Workflow",
        author="Tester",
        description="",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="1.0.0", category=category),
        nodes=[],
        edges=[],
        form=None,
    )

    return WorkflowRecordDTO(
        workflow_id=workflow_id,
        workflow=workflow,
        name=workflow.name,
        created_at="2026-04-08T00:00:00Z",
        updated_at="2026-04-08T00:00:00Z",
        opened_at=None,
        user_id=user_id,
        is_public=is_public,
    )


def build_user_dto(*, user_id: str = "user-1", is_admin: bool = False) -> UserDTO:
    return UserDTO(
        user_id=user_id,
        email=f"{user_id}@example.test",
        display_name=user_id,
        is_admin=is_admin,
        is_active=True,
        created_at="2026-04-08T00:00:00Z",
        updated_at="2026-04-08T00:00:00Z",
        last_login_at=None,
    )


def build_context(
    *,
    workflow_record: WorkflowRecordDTO | None = None,
    queue_user_id: str = "owner-1",
    multiuser: bool = False,
    user_is_admin: bool = False,
    workflow_not_found: bool = False,
):
    services = SimpleNamespace(
        configuration=SimpleNamespace(multiuser=multiuser),
        users=Mock(),
        workflow_records=Mock(),
    )
    services.users.get.return_value = build_user_dto(user_id=queue_user_id, is_admin=user_is_admin)

    if workflow_not_found:
        services.workflow_records.get.side_effect = WorkflowNotFoundError("missing")
    else:
        services.workflow_records.get.return_value = workflow_record or build_workflow_record_dto()

    context = Mock()
    context._services = services
    context._data = SimpleNamespace(queue_item=SimpleNamespace(user_id=queue_user_id))
    return context


def test_call_saved_workflows_invocation_contract():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation
    from invokeai.app.invocations.primitives import IntegerOutput

    invocation = CallSavedWorkflowsInvocation(id="test-node", workflow_id="workflow-123")

    assert invocation.get_type() == "call_saved_workflows"
    assert invocation.workflow_id == "workflow-123"

    output = invocation.invoke(build_context())

    assert isinstance(output, IntegerOutput)
    assert output.value == 0


def test_call_saved_workflows_invocation_raises_when_workflow_id_is_empty():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    invocation = CallSavedWorkflowsInvocation(id="test-node")

    with pytest.raises(ValueError, match="saved workflow must be selected"):
        invocation.invoke(build_context())


def test_call_saved_workflows_invocation_raises_when_workflow_does_not_exist():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    invocation = CallSavedWorkflowsInvocation(id="test-node", workflow_id="missing-workflow")

    with pytest.raises(ValueError, match="could not be found"):
        invocation.invoke(build_context(workflow_not_found=True))


def test_call_saved_workflows_invocation_raises_when_workflow_is_not_accessible():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    invocation = CallSavedWorkflowsInvocation(id="test-node", workflow_id="private-workflow")

    with pytest.raises(ValueError, match="is not accessible"):
        invocation.invoke(
            build_context(
                workflow_record=build_workflow_record_dto(
                    workflow_id="private-workflow",
                    user_id="owner-1",
                    category=WorkflowCategory.User,
                    is_public=False,
                ),
                queue_user_id="other-user",
                multiuser=True,
                user_is_admin=False,
            )
        )


def test_call_saved_workflows_invocation_allows_shared_workflow_for_non_owner():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    invocation = CallSavedWorkflowsInvocation(id="test-node", workflow_id="shared-workflow")

    output = invocation.invoke(
        build_context(
            workflow_record=build_workflow_record_dto(
                workflow_id="shared-workflow",
                user_id="owner-1",
                category=WorkflowCategory.User,
                is_public=True,
            ),
            queue_user_id="other-user",
            multiuser=True,
            user_is_admin=False,
        )
    )

    assert output.value == 0


def test_call_saved_workflows_invocation_allows_default_workflow_for_non_owner():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    invocation = CallSavedWorkflowsInvocation(id="test-node", workflow_id="default-workflow")

    output = invocation.invoke(
        build_context(
            workflow_record=build_workflow_record_dto(
                workflow_id="default-workflow",
                user_id="system",
                category=WorkflowCategory.Default,
                is_public=False,
            ),
            queue_user_id="other-user",
            multiuser=True,
            user_is_admin=False,
        )
    )

    assert output.value == 0


def test_call_saved_workflows_invocation_allows_admin_to_access_private_workflow():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    invocation = CallSavedWorkflowsInvocation(id="test-node", workflow_id="private-workflow")

    output = invocation.invoke(
        build_context(
            workflow_record=build_workflow_record_dto(
                workflow_id="private-workflow",
                user_id="owner-1",
                category=WorkflowCategory.User,
                is_public=False,
            ),
            queue_user_id="admin-user",
            multiuser=True,
            user_is_admin=True,
        )
    )

    assert output.value == 0


def test_call_saved_workflows_invocation_schema_hides_editor_managed_fields():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    schema = CallSavedWorkflowsInvocation.model_json_schema()
    workflow_id = schema["properties"]["workflow_id"]

    assert workflow_id["default"] == ""
    assert workflow_id["ui_hidden"] is True
    assert workflow_id["input"] == "any"
