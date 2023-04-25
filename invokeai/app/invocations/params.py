# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from .math import IntOutput

# Pass-through parameter nodes - used by subgraphs

class ParamIntInvocation(BaseInvocation):
    """An integer parameter"""
    #fmt: off
    type: Literal["param_int"] = "param_int"
    a: int = Field(default=0, description="The integer value")
    #fmt: on

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a)
