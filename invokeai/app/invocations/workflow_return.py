from typing import Any

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import Input, InputField, OutputField, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("workflow_return_output")
class WorkflowReturnOutput(BaseInvocationOutput):
    """The explicit named values returned from a callable workflow."""

    values: dict[str, Any] = OutputField(
        default={},
        description="The workflow return values, keyed by return name.",
        title="Values",
        ui_type=UIType.Any,
    )


class WorkflowReturnValueField(BaseModel):
    """One named workflow return value."""

    key: str = Field(description="The workflow return key.")
    value: Any = Field(description="The workflow return value.")


@invocation_output("workflow_return_value_output")
class WorkflowReturnValueOutput(BaseInvocationOutput):
    """A named workflow return value."""

    value: WorkflowReturnValueField = OutputField(
        description="The named workflow return value.",
        title="Return Value",
        ui_type=UIType._CollectionItem,
    )


@invocation(
    "workflow_return_value",
    title="Workflow Return Value",
    tags=["workflow", "return", "output"],
    category="workflow",
    version="1.0.0",
    classification=Classification.Beta,
    use_cache=False,
)
class WorkflowReturnValueInvocation(BaseInvocation):
    """Creates one named value for a callable workflow return."""

    key: str = InputField(default="", description="The return key.", title="Key")
    value: Any = InputField(
        default=None,
        description="The value returned under this key.",
        title="Value",
        ui_type=UIType.Any,
    )

    def invoke(self, context: InvocationContext) -> WorkflowReturnValueOutput:
        key = self.key.strip()
        if not key:
            raise ValueError("Workflow return key must not be empty.")
        return WorkflowReturnValueOutput(value=WorkflowReturnValueField(key=key, value=self.value))


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
    """Defines the explicit named result returned by a callable workflow."""

    values: WorkflowReturnValueField | list[WorkflowReturnValueField] = InputField(
        default=[],
        description="The named values returned to a calling workflow.",
        title="Values",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> WorkflowReturnOutput:
        named_values: dict[str, Any] = {}
        return_values = self.values if isinstance(self.values, list) else [self.values]
        for value in return_values:
            key = value.key.strip()
            if not key:
                raise ValueError("Workflow return key must not be empty.")
            if key in named_values:
                raise ValueError(f"Duplicate workflow return key '{key}'.")
            named_values[key] = value.value
        return WorkflowReturnOutput(values=named_values)


@invocation_output("workflow_return_get_output")
class WorkflowReturnGetOutput(BaseInvocationOutput):
    """A value extracted from named workflow return values."""

    value: Any = OutputField(description="The extracted workflow return value.", title="Value", ui_type=UIType.Any)


@invocation(
    "workflow_return_get",
    title="Get Workflow Return Value",
    tags=["workflow", "return", "input"],
    category="workflow",
    version="1.0.0",
    classification=Classification.Beta,
    use_cache=False,
)
class WorkflowReturnGetInvocation(BaseInvocation):
    """Extracts one named value from a callable workflow return."""

    values: dict[str, Any] = InputField(
        default={},
        description="The named workflow return values.",
        title="Values",
        ui_type=UIType.Any,
        input=Input.Connection,
    )
    key: str = InputField(default="", description="The return key to extract.", title="Key")

    def invoke(self, context: InvocationContext) -> WorkflowReturnGetOutput:
        key = self.key.strip()
        if not key:
            raise ValueError("Workflow return key must not be empty.")
        if key not in self.values:
            raise ValueError(f"Workflow return key '{key}' was not found.")
        return WorkflowReturnGetOutput(value=self.values[key])
