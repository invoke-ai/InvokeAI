from unittest.mock import Mock

from invokeai.app.services.shared.invocation_context import InvocationContext


def test_call_saved_workflows_invocation_contract():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation
    from invokeai.app.invocations.primitives import IntegerOutput

    invocation = CallSavedWorkflowsInvocation(id="test-node")

    assert invocation.get_type() == "call_saved_workflows"
    assert invocation.workflow_id == ""

    output = invocation.invoke(Mock(InvocationContext))

    assert isinstance(output, IntegerOutput)
    assert output.value == 0


def test_call_saved_workflows_invocation_schema_hides_editor_managed_fields():
    from invokeai.app.invocations.call_saved_workflows import CallSavedWorkflowsInvocation

    schema = CallSavedWorkflowsInvocation.model_json_schema()
    workflow_id = schema["properties"]["workflow_id"]

    assert workflow_id["default"] == ""
    assert workflow_id["ui_hidden"] is True
    assert workflow_id["input"] == "any"
