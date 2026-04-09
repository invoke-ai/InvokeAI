from typing import Any

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("workflow_return_output")
class WorkflowReturnOutput(BaseInvocationOutput):
    """The explicit collection returned from a callable workflow."""

    collection: list[Any] = OutputField(
        description="The workflow return collection",
        title="Collection",
        ui_type=UIType._Collection,
    )


@invocation(
    "workflow_return",
    title="Workflow Return",
    tags=["workflow", "return", "output"],
    category="workflow",
    version="1.0.0",
    classification=Classification.Beta,
    use_cache=False,
)
class WorkflowReturnInvocation(BaseInvocation):
    """Defines the explicit collection result returned by a callable workflow."""

    collection: list[Any] = InputField(
        default=[],
        description="The collection returned to a calling workflow.",
        title="Collection",
        ui_type=UIType._Collection,
    )

    def invoke(self, context: InvocationContext) -> WorkflowReturnOutput:
        return WorkflowReturnOutput(collection=self.collection)
