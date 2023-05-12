# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

from pydantic import BaseModel, Field
import numpy as np

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig


class MathInvocationConfig(BaseModel):
    """Helper class to provide all math invocations with additional config"""

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["math"],
            }
        }


class IntOutput(BaseInvocationOutput):
    """An integer output"""
    #fmt: off
    type: Literal["int_output"] = "int_output"
    a: int = Field(default=None, description="The output integer")
    #fmt: on


class AddInvocation(BaseInvocation, MathInvocationConfig):
    """Adds two numbers"""
    #fmt: off
    type: Literal["add"] = "add"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a + self.b)


class SubtractInvocation(BaseInvocation, MathInvocationConfig):
    """Subtracts two numbers"""
    #fmt: off
    type: Literal["sub"] = "sub"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a - self.b)


class MultiplyInvocation(BaseInvocation, MathInvocationConfig):
    """Multiplies two numbers"""
    #fmt: off
    type: Literal["mul"] = "mul"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a * self.b)


class DivideInvocation(BaseInvocation, MathInvocationConfig):
    """Divides two numbers"""
    #fmt: off
    type: Literal["div"] = "div"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=int(self.a / self.b))


class RandomIntInvocation(BaseInvocation):
    """Outputs a single random integer."""
    #fmt: off
    type: Literal["rand_int"] = "rand_int"
    #fmt: on
    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=np.random.randint(0, np.iinfo(np.int32).max))
