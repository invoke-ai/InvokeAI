# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

import numpy as np
from typing import Literal

from invokeai.app.invocations.primitives import IntegerOutput, FloatOutput
from pydantic import validator

from .baseinvocation import BaseInvocation, FieldDescriptions, InputField, InvocationContext, invocation


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


@invocation("rand_int", title="Random Integer", tags=["math", "random"], category="math", version="1.0.0")
class RandomIntInvocation(BaseInvocation):
    """Outputs a single random integer."""

    low: int = InputField(default=0, description="The inclusive low value")
    high: int = InputField(default=np.iinfo(np.int32).max, description="The exclusive high value")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=np.random.randint(self.low, self.high))


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
    "Add A+B",
    "Subtract A-B",
    "Multiply A*B",
    "Divide A/B",
    "Exponentiate A^B",
    "Modulus A%B",
    "Absolute Value of A",
    "Minimum(A,B)",
    "Maximum(A,B)",
]


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

    operation: INTEGER_OPERATIONS = InputField(default="Add A+B", description="The operation to perform")
    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    @validator("b")
    def no_unrepresentable_results(cls, v, values):
        if values["operation"] == "Divide A/B" and v == 0:
            raise ValueError("Cannot divide by zero")
        elif values["operation"] == "Modulus A%B" and v == 0:
            raise ValueError("Cannot divide by zero")
        elif values["operation"] == "Exponentiate A^B" and v < 0:
            raise ValueError("Result of exponentiation is not an integer")
        return v

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        # Python doesn't support switch statements until 3.10, but InvokeAI supports back to 3.9
        if self.operation == "Add A+B":
            return IntegerOutput(value=self.a + self.b)
        elif self.operation == "Subtract A-B":
            return IntegerOutput(value=self.a - self.b)
        elif self.operation == "Multiply A*B":
            return IntegerOutput(value=self.a * self.b)
        elif self.operation == "Divide A/B":
            return IntegerOutput(value=int(self.a / self.b))
        elif self.operation == "Exponentiate A^B":
            return IntegerOutput(value=self.a**self.b)
        elif self.operation == "Modulus A%B":
            return IntegerOutput(value=self.a % self.b)
        elif self.operation == "Absolute Value of A":
            return IntegerOutput(value=abs(self.a))
        elif self.operation == "Minimum(A,B)":
            return IntegerOutput(value=min(self.a, self.b))
        else:  # self.operation == "Maximum(A,B)":
            return IntegerOutput(value=max(self.a, self.b))


FLOAT_OPERATIONS = Literal[
    "Add A+B",
    "Subtract A-B",
    "Multiply A*B",
    "Divide A/B",
    "Exponentiate A^B",
    "Absolute Value of A",
    "Minimum(A,B)",
    "Maximum(A,B)",
]


@invocation(
    "float_math",
    title="Float Math",
    tags=["math", "float", "add", "subtract", "multiply", "divide", "power", "root", "absolute value", "min", "max"],
    category="math",
    version="1.0.0",
)
class FloatMathInvocation(BaseInvocation):
    """Performs floating point math."""

    operation: FLOAT_OPERATIONS = InputField(default="Add A+B", description="The operation to perform")
    a: float = InputField(default=0, description=FieldDescriptions.num_1)
    b: float = InputField(default=0, description=FieldDescriptions.num_2)

    @validator("b")
    def no_unrepresentable_results(cls, v, values):
        if values["operation"] == "Divide A/B" and v == 0:
            raise ValueError("Cannot divide by zero")
        elif values["operation"] == "Exponentiate A^B" and values["a"] == 0 and v < 0:
            raise ValueError("Cannot raise zero to a negative power")
        elif values["operation"] == "Exponentiate A^B" and type(values["a"] ** v) == complex:
            raise ValueError("Root operation resulted in a complex number")
        return v

    def invoke(self, context: InvocationContext) -> FloatOutput:
        # Python doesn't support switch statements until 3.10, but InvokeAI supports back to 3.9
        if self.operation == "Add A+B":
            return FloatOutput(value=self.a + self.b)
        elif self.operation == "Subtract A-B":
            return FloatOutput(value=self.a - self.b)
        elif self.operation == "Multiply A*B":
            return FloatOutput(value=self.a * self.b)
        elif self.operation == "Divide A/B":
            return FloatOutput(value=self.a / self.b)
        elif self.operation == "Exponentiate A^B":
            return FloatOutput(value=self.a**self.b)
        elif self.operation == "Square Root of A":
            return FloatOutput(value=np.sqrt(self.a))
        elif self.operation == "Absolute Value of A":
            return FloatOutput(value=abs(self.a))
        elif self.operation == "Minimum(A,B)":
            return FloatOutput(value=min(self.a, self.b))
        else:  # self.operation == "Maximum(A,B)":
            return FloatOutput(value=max(self.a, self.b))
