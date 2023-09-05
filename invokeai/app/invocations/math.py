# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

import numpy as np

from invokeai.app.invocations.primitives import IntegerOutput

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
