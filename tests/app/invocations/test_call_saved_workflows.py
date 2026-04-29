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
    WorkflowWithoutIDValidator,
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
    user_exists: bool = True,
    workflow_not_found: bool = False,
):
    services = SimpleNamespace(
        configuration=SimpleNamespace(multiuser=multiuser),
        users=Mock(),
        workflow_records=Mock(),
    )
    services.users.get.return_value = (
        build_user_dto(user_id=queue_user_id, is_admin=user_is_admin) if user_exists else None
    )

    if workflow_not_found:
        services.workflow_records.get.side_effect = WorkflowNotFoundError("missing")
    else:
        services.workflow_records.get.return_value = workflow_record or build_workflow_record_dto()

    context = Mock()
    context._services = services
    context._data = SimpleNamespace(queue_item=SimpleNamespace(user_id=queue_user_id))
    return context


def test_call_saved_workflow_invocation_contract():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
    from invokeai.app.invocations.workflow_return import WorkflowReturnOutput

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="workflow-123")

    assert invocation.get_type() == "call_saved_workflow"
    assert invocation.workflow_id == "workflow-123"

    output = invocation.invoke(build_context())

    assert isinstance(output, WorkflowReturnOutput)
    assert output.values == {}


def test_call_saved_workflow_invocation_raises_when_workflow_id_is_empty():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node")

    with pytest.raises(ValueError, match="saved workflow must be selected"):
        invocation.invoke(build_context())


def test_call_saved_workflow_invocation_raises_when_workflow_does_not_exist():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="missing-workflow")

    with pytest.raises(ValueError, match="could not be found"):
        invocation.invoke(build_context(workflow_not_found=True))


def test_call_saved_workflow_invocation_raises_when_workflow_is_not_accessible():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="private-workflow")

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


def test_call_saved_workflow_invocation_allows_shared_workflow_for_non_owner():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="shared-workflow")

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

    assert output.values == {}


def test_call_saved_workflow_invocation_allows_default_workflow_for_non_owner():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="default-workflow")

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

    assert output.values == {}


def test_call_saved_workflow_invocation_allows_admin_to_access_private_workflow():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="private-workflow")

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

    assert output.values == {}


def test_call_saved_workflow_invocation_raises_when_private_workflow_user_record_is_missing():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    invocation = CallSavedWorkflowInvocation(id="test-node", workflow_id="private-workflow")

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
                user_exists=False,
            )
        )


def test_call_saved_workflow_invocation_schema_declares_saved_workflow_ui_type():
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation

    schema = CallSavedWorkflowInvocation.model_json_schema()
    workflow_id = schema["properties"]["workflow_id"]
    workflow_inputs = schema["properties"]["workflow_inputs"]

    assert workflow_id["default"] == ""
    assert workflow_id["input"] == "any"
    assert workflow_id["ui_type"] == "SavedWorkflowField"
    assert workflow_inputs["default"] == {}
    assert workflow_inputs["ui_hidden"] is True


def test_workflow_return_invocation_contract():
    from invokeai.app.invocations.workflow_return import (
        WorkflowReturnInvocation,
        WorkflowReturnOutput,
        WorkflowReturnValueField,
    )

    invocation = WorkflowReturnInvocation(
        id="return-node",
        values=[
            WorkflowReturnValueField(key="prompt", value="a"),
            WorkflowReturnValueField(key="count", value=1),
            WorkflowReturnValueField(key="metadata", value={"x": True}),
        ],
    )

    assert invocation.get_type() == "workflow_return"

    output = invocation.invoke(build_context())

    assert isinstance(output, WorkflowReturnOutput)
    assert output.values == {"prompt": "a", "count": 1, "metadata": {"x": True}}
    assert not hasattr(output, "collection")


def test_workflow_return_invocation_accepts_single_return_value():
    from invokeai.app.invocations.workflow_return import WorkflowReturnInvocation, WorkflowReturnValueField

    invocation = WorkflowReturnInvocation(id="return-node", values=WorkflowReturnValueField(key="sum", value=3))

    output = invocation.invoke(build_context())

    assert output.values == {"sum": 3}


def test_workflow_return_values_schema_preserves_single_or_list_cardinality():
    from invokeai.app.invocations.workflow_return import WorkflowReturnInvocation

    values_schema = WorkflowReturnInvocation.model_json_schema()["properties"]["values"]

    assert values_schema["anyOf"] == [
        {"$ref": "#/$defs/WorkflowReturnValueField"},
        {"items": {"$ref": "#/$defs/WorkflowReturnValueField"}, "type": "array"},
    ]
    assert values_schema.get("ui_type") != "CollectionField"


def test_workflow_return_value_invocation_contract():
    from invokeai.app.invocations.workflow_return import WorkflowReturnValueField, WorkflowReturnValueInvocation

    invocation = WorkflowReturnValueInvocation(id="return-value-node", key="image", value={"image_name": "image-a"})

    output = invocation.invoke(build_context())

    assert output.value == WorkflowReturnValueField(key="image", value={"image_name": "image-a"})


def test_workflow_return_invocation_rejects_duplicate_keys():
    from invokeai.app.invocations.workflow_return import WorkflowReturnInvocation, WorkflowReturnValueField

    invocation = WorkflowReturnInvocation(
        id="return-node",
        values=[
            WorkflowReturnValueField(key="image", value="image-a"),
            WorkflowReturnValueField(key="image", value="image-b"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate workflow return key 'image'"):
        invocation.invoke(build_context())


def test_workflow_return_get_invocation_contract():
    from invokeai.app.invocations.workflow_return import WorkflowReturnGetInvocation

    invocation = WorkflowReturnGetInvocation(id="return-get-node", values={"image": "image-a"}, key="image")

    output = invocation.invoke(build_context())

    assert output.value == "image-a"


def test_workflow_return_get_invocation_rejects_missing_key():
    from invokeai.app.invocations.workflow_return import WorkflowReturnGetInvocation

    invocation = WorkflowReturnGetInvocation(id="return-get-node", values={"image": "image-a"}, key="mask")

    with pytest.raises(ValueError, match="Workflow return key 'mask' was not found"):
        invocation.invoke(build_context())


def test_workflow_return_get_invocation_rejects_empty_key():
    from invokeai.app.invocations.workflow_return import WorkflowReturnGetInvocation

    invocation = WorkflowReturnGetInvocation(id="return-get-node", values={"image": "image-a"}, key=" ")

    with pytest.raises(ValueError, match="Workflow return key must not be empty"):
        invocation.invoke(build_context())


def test_workflow_return_invocation_schema_declares_named_values_contract():
    from invokeai.app.invocations.workflow_return import WorkflowReturnGetInvocation, WorkflowReturnInvocation

    schema = WorkflowReturnInvocation.model_json_schema()
    assert "collection" not in schema["properties"]
    values = schema["properties"]["values"]

    assert values["input"] == "connection"
    assert "ui_type" not in values

    get_schema = WorkflowReturnGetInvocation.model_json_schema()
    get_values = get_schema["properties"]["values"]
    assert get_values["input"] == "connection"
    assert get_values["ui_type"] == "AnyField"


def test_workflow_without_id_validator_rejects_duplicate_workflow_return_nodes():
    with pytest.raises(ValueError, match="workflow_return"):
        WorkflowWithoutIDValidator.validate_python(
            {
                "name": "Workflow With Duplicate Returns",
                "author": "Tester",
                "description": "",
                "version": "1.0.0",
                "contact": "",
                "tags": "",
                "notes": "",
                "exposedFields": [],
                "meta": {"version": "1.0.0", "category": "user"},
                "nodes": [
                    {
                        "id": "return-1",
                        "type": "invocation",
                        "data": {"id": "return-1", "type": "workflow_return"},
                        "position": {"x": 0, "y": 0},
                    },
                    {
                        "id": "return-2",
                        "type": "invocation",
                        "data": {"id": "return-2", "type": "workflow_return"},
                        "position": {"x": 100, "y": 0},
                    },
                ],
                "edges": [],
                "form": None,
            }
        )
