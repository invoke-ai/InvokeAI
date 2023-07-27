# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

from pydantic import Field

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext
from .math import FloatOutput, IntOutput

# Pass-through parameter nodes - used by subgraphs


class ParamIntInvocation(BaseInvocation):
    """An integer parameter"""

    # fmt: off
    type: Literal["param_int"] = "param_int"
    a: int = Field(default=0, description="The integer value")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"tags": ["param", "integer"], "title": "Integer Parameter"},
        }

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a)


class ParamFloatInvocation(BaseInvocation):
    """A float parameter"""

    # fmt: off
    type: Literal["param_float"] = "param_float"
    param: float = Field(default=0.0, description="The float value")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"tags": ["param", "float"], "title": "Float Parameter"},
        }

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(param=self.param)


class StringOutput(BaseInvocationOutput):
    """A string output"""

    type: Literal["string_output"] = "string_output"
    text: str = Field(default=None, description="The output string")


class ParamStringInvocation(BaseInvocation):
    """A string parameter"""

    type: Literal["param_string"] = "param_string"
    text: str = Field(default="", description="The string value")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"tags": ["param", "string"], "title": "String Parameter"},
        }

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(text=self.text)
