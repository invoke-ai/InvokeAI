# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

import numpy as np

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    InputField,
    InvocationContext,
    OutputField,
    tags,
    title,
)


class IntOutput(BaseInvocationOutput):
    """An integer output"""

    type: Literal["int_output"] = "int_output"
    a: int = OutputField(default=None, description="The output integer")


class FloatOutput(BaseInvocationOutput):
    """A float output"""

    type: Literal["float_output"] = "float_output"
    a: float = OutputField(default=None, description="The output float")


@title("Add Integers")
@tags("math")
class AddInvocation(BaseInvocation):
    """Adds two numbers"""

    type: Literal["add"] = "add"

    # Inputs
    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a + self.b)


@title("Subtract Integers")
@tags("math")
class SubtractInvocation(BaseInvocation):
    """Subtracts two numbers"""

    type: Literal["sub"] = "sub"

    # Inputs
    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a - self.b)


@title("Multiply Integers")
@tags("math")
class MultiplyInvocation(BaseInvocation):
    """Multiplies two numbers"""

    type: Literal["mul"] = "mul"

    # Inputs
    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a * self.b)


@title("Divide Integers")
@tags("math")
class DivideInvocation(BaseInvocation):
    """Divides two numbers"""

    type: Literal["div"] = "div"

    # Inputs
    a: int = InputField(default=0, description=FieldDescriptions.num_1)
    b: int = InputField(default=0, description=FieldDescriptions.num_2)

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=int(self.a / self.b))


@title("Random Integer")
@tags("math")
class RandomIntInvocation(BaseInvocation):
    """Outputs a single random integer."""

    type: Literal["rand_int"] = "rand_int"

    # Inputs
    low: int = InputField(default=0, description="The inclusive low value")
    high: int = InputField(default=np.iinfo(np.int32).max, description="The exclusive high value")

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=np.random.randint(self.low, self.high))
