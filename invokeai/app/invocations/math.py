# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from datetime import datetime, timezone
from typing import Literal, Optional

import numpy
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from ..services.image_storage import ImageType
from ..services.invocation_services import InvocationServices
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext


class IntOutput(BaseInvocationOutput):
    """An integer output"""
    #fmt: off
    type: Literal["int_output"] = "int_output"
    a: int = Field(default=None, description="The output integer")
    #fmt: on


class AddInvocation(BaseInvocation):
    """Adds two numbers"""
    #fmt: off
    type: Literal["add"] = "add"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a + self.b)


class SubtractInvocation(BaseInvocation):
    """Subtracts two numbers"""
    #fmt: off
    type: Literal["sub"] = "sub"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a - self.b)


class MultiplyInvocation(BaseInvocation):
    """Multiplies two numbers"""
    #fmt: off
    type: Literal["mul"] = "mul"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a * self.b)


class DivideInvocation(BaseInvocation):
    """Divides two numbers"""
    #fmt: off
    type: Literal["div"] = "div"
    a: int = Field(default=0, description="The first number")
    b: int = Field(default=0, description="The second number")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=int(self.a / self.b))
