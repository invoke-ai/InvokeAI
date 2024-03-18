# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

import numpy as np
from pydantic import ValidationInfo, field_validator

from invokeai.app.invocations.fields import FieldDescriptions, InputField
from invokeai.app.invocations.primitives import FloatOutput, IntegerOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

from .baseinvocation import BaseInvocation, invocation


@invocation("add", title="Add Integers", tags=["math", "add"], category="math", version="1.0.0")
class AddInvocation(BaseInvocation):
    """Adds two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a + self.b)


@invocation("sub", title="Subtract Integers", tags=["math", "subtract"], category="math", version="1.0.0")
class SubtractInvocation(BaseInvocation):
    """Subtracts two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a - self.b)


@invocation("mul", title="Multiply Integers", tags=["math", "multiply"], category="math", version="1.0.0")
class MultiplyInvocation(BaseInvocation):
    """Multiplies two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a * self.b)


@invocation("div", title="Divide Integers", tags=["math", "divide"], category="math", version="1.0.0")
class DivideInvocation(BaseInvocation):
    """Divides two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=int(self.a / self.b))


@invocation(
    "rand_int",
    title="Random Integer",
    tags=["math", "random"],
    category="math",
    version="1.0.0",
    use_cache=False,
)
class RandomIntInvocation(BaseInvocation):
    """Outputs a single random integer."""

    low: int = InputField(default=0, description=FieldDescriptions.inclusive_low)
    high: int = InputField(default=np.iinfo(np.int32).max, description=FieldDescriptions.exclusive_high)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=np.random.randint(self.low, self.high))


@invocation(
    "rand_float",
    title="Random Float",
    tags=["math", "float", "random"],
    category="math",
    version="1.0.1",
    use_cache=False,
)
class RandomFloatInvocation(BaseInvocation):
    """Outputs a single random float"""

    low: float = InputField(default=0.0, description=FieldDescriptions.inclusive_low)
    high: float = InputField(default=1.0, description=FieldDescriptions.exclusive_high)
    decimals: int = InputField(default=2, description=FieldDescriptions.decimal_places)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        random_float = np.random.uniform(self.low, self.high)
        rounded_float = round(random_float, self.decimals)
        return FloatOutput(value=rounded_float)


@invocation(
    "float_to_int",
    title="Float To Integer",
    tags=["math", "round", "integer", "float", "convert"],
    category="math",
    version="1.0.0",
)
class FloatToIntegerInvocation(BaseInvocation):
    """Rounds a float number to (a multiple of) an integer."""

    value: float = InputField(default=0, description="The value to round")
    multiple: int = InputField(default=1, ge=1, title="Multiple of", description="The multiple to round to")
    method: Literal["Nearest", "Floor", "Ceiling", "Truncate"] = InputField(
        default="Nearest", description="The method to use for rounding"
    )

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        if self.method == "Nearest":
            return IntegerOutput(value=round(self.value / self.multiple) * self.multiple)
        elif self.method == "Floor":
            return IntegerOutput(value=np.floor(self.value / self.multiple) * self.multiple)
        elif self.method == "Ceiling":
            return IntegerOutput(value=np.ceil(self.value / self.multiple) * self.multiple)
        else:  # self.method == "Truncate"
            return IntegerOutput(value=int(self.value / self.multiple) * self.multiple)


@invocation("round_float", title="Round Float", tags=["math", "round"], category="math", version="1.0.0")
class RoundInvocation(BaseInvocation):
    """Rounds a float to a specified number of decimal places."""

    value: float = InputField(default=0, description="The float value")
    decimals: int = InputField(default=0, description="The number of decimal places")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=round(self.value, self.decimals))


INTEGER_OPERATIONS = Literal[
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "EXP",
    "MOD",
    "ABS",
    "MIN",
    "MAX",
]


INTEGER_OPERATIONS_LABELS = {
    "ADD": "Add A+B",
    "SUB": "Subtract A-B",
    "MUL": "Multiply A*B",
    "DIV": "Divide A/B",
    "EXP": "Exponentiate A^B",
    "MOD": "Modulus A%B",
    "ABS": "Absolute Value of A",
    "MIN": "Minimum(A,B)",
    "MAX": "Maximum(A,B)",
}


@invocation(
    "integer_math",
    title="Integer Math",
    tags=[
        "math",
        "integer",
        "add",
        "subtract",
        "multiply",
        "divide",
        "modulus",
        "power",
        "absolute value",
        "min",
        "max",
    ],
    category="math",
    version="1.0.0",
)
class IntegerMathInvocation(BaseInvocation):
    """Performs integer math."""

    operation: INTEGER_OPERATIONS = InputField(
        default="ADD", description="The operation to perform", ui_choice_labels=INTEGER_OPERATIONS_LABELS
    )
    a: int = InputField(default=1, description=FieldDescriptions.num_1)
    b: int = InputField(default=1, description=FieldDescriptions.num_2)

    @field_validator("b")
    def no_unrepresentable_results(cls, v: int, info: ValidationInfo):
        if info.data["operation"] == "DIV" and v == 0:
            raise ValueError("Cannot divide by zero")
        elif info.data["operation"] == "MOD" and v == 0:
            raise ValueError("Cannot divide by zero")
        elif info.data["operation"] == "EXP" and v < 0:
            raise ValueError("Result of exponentiation is not an integer")
        return v

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        # Python doesn't support switch statements until 3.10, but InvokeAI supports back to 3.9
        if self.operation == "ADD":
            return IntegerOutput(value=self.a + self.b)
        elif self.operation == "SUB":
            return IntegerOutput(value=self.a - self.b)
        elif self.operation == "MUL":
            return IntegerOutput(value=self.a * self.b)
        elif self.operation == "DIV":
            return IntegerOutput(value=int(self.a / self.b))
        elif self.operation == "EXP":
            return IntegerOutput(value=self.a**self.b)
        elif self.operation == "MOD":
            return IntegerOutput(value=self.a % self.b)
        elif self.operation == "ABS":
            return IntegerOutput(value=abs(self.a))
        elif self.operation == "MIN":
            return IntegerOutput(value=min(self.a, self.b))
        else:  # self.operation == "MAX":
            return IntegerOutput(value=max(self.a, self.b))


FLOAT_OPERATIONS = Literal[
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "EXP",
    "ABS",
    "SQRT",
    "MIN",
    "MAX",
]


FLOAT_OPERATIONS_LABELS = {
    "ADD": "Add A+B",
    "SUB": "Subtract A-B",
    "MUL": "Multiply A*B",
    "DIV": "Divide A/B",
    "EXP": "Exponentiate A^B",
    "ABS": "Absolute Value of A",
    "SQRT": "Square Root of A",
    "MIN": "Minimum(A,B)",
    "MAX": "Maximum(A,B)",
}


@invocation(
    "float_math",
    title="Float Math",
    tags=["math", "float", "add", "subtract", "multiply", "divide", "power", "root", "absolute value", "min", "max"],
    category="math",
    version="1.0.0",
)
class FloatMathInvocation(BaseInvocation):
    """Performs floating point math."""

    operation: FLOAT_OPERATIONS = InputField(
        default="ADD", description="The operation to perform", ui_choice_labels=FLOAT_OPERATIONS_LABELS
    )
    a: float = InputField(default=1, description=FieldDescriptions.num_1)
    b: float = InputField(default=1, description=FieldDescriptions.num_2)

    @field_validator("b")
    def no_unrepresentable_results(cls, v: float, info: ValidationInfo):
        if info.data["operation"] == "DIV" and v == 0:
            raise ValueError("Cannot divide by zero")
        elif info.data["operation"] == "EXP" and info.data["a"] == 0 and v < 0:
            raise ValueError("Cannot raise zero to a negative power")
        elif info.data["operation"] == "EXP" and isinstance(info.data["a"] ** v, complex):
            raise ValueError("Root operation resulted in a complex number")
        return v

    def invoke(self, context: InvocationContext) -> FloatOutput:
        # Python doesn't support switch statements until 3.10, but InvokeAI supports back to 3.9
        if self.operation == "ADD":
            return FloatOutput(value=self.a + self.b)
        elif self.operation == "SUB":
            return FloatOutput(value=self.a - self.b)
        elif self.operation == "MUL":
            return FloatOutput(value=self.a * self.b)
        elif self.operation == "DIV":
            return FloatOutput(value=self.a / self.b)
        elif self.operation == "EXP":
            return FloatOutput(value=self.a**self.b)
        elif self.operation == "SQRT":
            return FloatOutput(value=np.sqrt(self.a))
        elif self.operation == "ABS":
            return FloatOutput(value=abs(self.a))
        elif self.operation == "MIN":
            return FloatOutput(value=min(self.a, self.b))
        else:  # self.operation == "MAX":
            return FloatOutput(value=max(self.a, self.b))
