from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.primitives import IntegerOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "call_saved_workflows",
    title="Call Saved Workflows",
    tags=["workflow", "saved", "library"],
    category="workflow",
    version="1.0.0",
    use_cache=False,
    classification=Classification.Beta,
)
class CallSavedWorkflowsInvocation(BaseInvocation):
    """Displays and later executes against the saved workflow library."""

    workflow_id: str = InputField(
        default="",
        description="The selected saved workflow ID, managed by the workflow editor UI.",
        ui_hidden=True,
    )

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=0)
