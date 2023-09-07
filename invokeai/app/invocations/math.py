# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

import numpy as np
from typing import Literal

from invokeai.app.invocations.primitives import IntegerOutput, FloatOutput

from .baseinvocation import BaseInvocation, FieldDescriptions, InputField, InvocationContext, invocation


@invocation("add", title="Add Integers", tags=["math", "add"], category="math", version="1.0.0")
class AddInvocation(BaseInvocation):
    """Adds two integer numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a + self.b)
    
@invocation("add_float", title="Add Floats", tags=["math", "add", "float"], category="math", version="1.0.0")
class AddFloatInvocation(BaseInvocation):
    """Adds two float numbers"""

    a: float = InputField(default=0, description=FieldDescriptions.num_1)
    b: float = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=self.a + self.b)


@invocation("sub", title="Subtract Integers", tags=["math", "subtract"], category="math", version="1.0.0")
class SubtractInvocation(BaseInvocation):
    """Subtracts two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a - self.b)
    
@invocation("sub_float", title="Subtract Floats", tags=["math", "subtract", "float"], category="math", version="1.0.0")
class SubtractFloatInvocation(BaseInvocation):
    """Subtracts two float numbers"""

    a: float = InputField(default=0, description=FieldDescriptions.num_1)
    b: float = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=self.a - self.b)


@invocation("mul", title="Multiply Integers", tags=["math", "multiply"], category="math", version="1.0.0")
class MultiplyInvocation(BaseInvocation):
    """Multiplies two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a * self.b)

@invocation("mul_float", title="Multiply Floats", tags=["math", "multiply", "float"], category="math", version="1.0.0")
class MultiplyFloatInvocation(BaseInvocation):
    """Multiplies two float numbers"""

    a: float = InputField(default=0, description=FieldDescriptions.num_1)
    b: float = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=self.a * self.b)


@invocation("div", title="Divide Integers", tags=["math", "divide"], category="math", version="1.0.0")
class DivideInvocation(BaseInvocation):
    """Divides two numbers"""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=int(self.a / self.b))

@invocation("div_float", title="Divide Floats", tags=["math", "divide", "float"], category="math", version="1.0.0")
class DivideFloatInvocation(BaseInvocation):
    """Divides two float numbers"""

    a: float = InputField(default=0, description=FieldDescriptions.num_1)
    b: float = InputField(default=1, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=self.a / self.b)


@invocation("rand_int", title="Random Integer", tags=["math", "random"], category="math", version="1.0.0")
class RandomIntInvocation(BaseInvocation):
    """Outputs a single random integer."""

    low: int = InputField(default=0, description="The inclusive low value")
    high: int = InputField(default=np.iinfo(np.int32).max, description="The exclusive high value")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=np.random.randint(self.low, self.high))


@invocation("round_to_multiple", title="Round to Multiple", tags=["math", "round", "integer", "convert"], category="math", version="1.0.0")
class RoundToMultipleInvocation(BaseInvocation):
    """Rounds a number to the nearest integer multiple."""

    value: float = InputField(default=0, description="The value to round")
    multiple: int = InputField(default=1, ge=1, description="The multiple to round to")
    method: Literal["Nearest", "Floor", "Ceiling"] = InputField(default="Nearest", description="The method to use for rounding")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        if self.method == "Nearest":
            return IntegerOutput(value=round(self.value / self.multiple) * self.multiple)
        elif self.method == "Floor":
            return IntegerOutput(value=np.floor(self.value / self.multiple) * self.multiple)
        else: #self.method == "Ceiling"
            return IntegerOutput(value=np.ceil(self.value / self.multiple) * self.multiple)


@invocation("round_float", title="Round Float", tags=["math", "round"], category="math", version="1.0.0")
class RoundInvocation(BaseInvocation):
    """Rounds a float to a specified number of decimal places."""

    value: float = InputField(default=0, description="The float value")
    decimals: int = InputField(default=0, description="The number of decimal places")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=round(self.value, self.decimals))


@invocation("abs", title="Absolute Value", tags=["math", "abs"], category="math", version="1.0.0")
class AbsoluteValueInvocation(BaseInvocation):
    """Returns the absolute value of a number."""

    value: float = InputField(default=0, description="The float value")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=abs(self.value))


@invocation("mod", title="Modulus", tags=["math", "modulus"], category="math", version="1.0.0")
class ModulusInvocation(BaseInvocation):
    """Returns the modulus of two numbers."""

    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(value=self.a % self.b)


@invocation("sqrt", title="Square Root", tags=["math", "sqrt"], category="math", version="1.0.0")
class SquareRootInvocation(BaseInvocation):
    """Returns the square root of a number."""

    value: float = InputField(default=0, ge=0, description="The float value")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(value=np.sqrt(self.value))