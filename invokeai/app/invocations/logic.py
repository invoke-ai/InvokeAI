from typing import Any, Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("if_output")
class IfInvocationOutput(BaseInvocationOutput):
    value: Optional[Any] = OutputField(
        default=None, description="The selected value", title="Output", ui_type=UIType.Any
    )


@invocation("if", title="If", tags=["logic", "conditional"], category="logic", version="1.0.0")
class IfInvocation(BaseInvocation):
    """Selects between two optional inputs based on a boolean condition."""

    condition: bool = InputField(default=False, description="The condition used to select an input", title="Condition")
    true_input: Optional[Any] = InputField(
        default=None,
        description="Selected when the condition is true",
        title="True Input",
        ui_type=UIType.Any,
    )
    false_input: Optional[Any] = InputField(
        default=None,
        description="Selected when the condition is false",
        title="False Input",
        ui_type=UIType.Any,
    )

    def invoke(self, context: InvocationContext) -> IfInvocationOutput:
        return IfInvocationOutput(value=self.true_input if self.condition else self.false_input)
