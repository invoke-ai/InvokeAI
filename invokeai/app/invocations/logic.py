from typing import Any, Literal, Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField, UIType
from invokeai.app.invocations.primitives import BooleanOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("if_output")
class IfInvocationOutput(BaseInvocationOutput):
    value: Optional[Any] = OutputField(
        default=None, description="The selected value", title="Output", ui_type=UIType.Any
    )


@invocation("if", title="If", tags=["logic", "conditional"], category="math", version="1.0.0")
class IfInvocation(BaseInvocation):
    """Selects between two optional inputs based on a boolean condition."""

    execution_effects_enabled = True
    execution_activation_fields = frozenset({"true_input", "false_input"})

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
        selected_field = "true_input" if self.condition else "false_input"
        execution = getattr(context, "execution", None)
        if execution is not None:
            execution.emit(selected_field, selected_field, token_kind="activation")
        return IfInvocationOutput(value=self.true_input if self.condition else self.false_input)


BOOLEAN_OPERATIONS = Literal["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT"]


@invocation("boolean_logic", title="Boolean Logic", tags=["logic", "boolean"], category="math", version="1.0.0")
class BooleanLogicInvocation(BaseInvocation):
    """Performs Boolean AND, OR, XOR, NOT, NAND, NOR, or XNOR operations. NOT uses only A."""

    operation: BOOLEAN_OPERATIONS = InputField(
        default="AND",
        description="The logical operation to perform",
        ui_choice_labels={
            "AND": "A AND B",
            "OR": "A OR B",
            "XOR": "A XOR B",
            "NAND": "A NAND B",
            "NOR": "A NOR B",
            "XNOR": "A XNOR B",
            "NOT": "NOT A",
        },
    )
    a: bool = InputField(default=False, description="First Boolean input")
    b: bool = InputField(default=False, description="Second Boolean input (ignored for NOT)")

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        if self.operation == "AND":
            result = self.a and self.b
        elif self.operation == "OR":
            result = self.a or self.b
        elif self.operation == "XOR":
            result = self.a != self.b
        elif self.operation == "NAND":
            result = not (self.a and self.b)
        elif self.operation == "NOR":
            result = not (self.a or self.b)
        elif self.operation == "XNOR":
            result = self.a == self.b
        else:  # self.operation == "NOT":
            result = not self.a
        return BooleanOutput(value=result)
