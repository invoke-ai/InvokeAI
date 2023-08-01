# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

from invokeai.app.invocations.prompt import PromptOutput

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    tags,
    title,
)
from .math import FloatOutput, IntOutput

# Pass-through parameter nodes - used by subgraphs


@title("Integer Parameter")
@tags("integer")
class ParamIntInvocation(BaseInvocation):
    """An integer parameter"""

    type: Literal["param_int"] = "param_int"

    # Inputs
    a: int = InputField(default=0, description="The integer value")

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a)


@title("Float Parameter")
@tags("float")
class ParamFloatInvocation(BaseInvocation):
    """A float parameter"""

    type: Literal["param_float"] = "param_float"

    # Inputs
    param: float = InputField(default=0.0, description="The float value")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(a=self.param)


class StringOutput(BaseInvocationOutput):
    """A string output"""

    type: Literal["string_output"] = "string_output"
    text: str = OutputField(description="The output string")


@title("String Parameter")
@tags("string")
class ParamStringInvocation(BaseInvocation):
    """A string parameter"""

    type: Literal["param_string"] = "param_string"

    # Inputs
    text: str = InputField(default="", description="The string value")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(text=self.text)


@title("Prompt Parameter")
@tags("prompt")
class ParamPromptInvocation(BaseInvocation):
    """A prompt input parameter"""

    type: Literal["param_prompt"] = "param_prompt"

    # Inputs
    prompt: str = InputField(default="", description="The prompt value")

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(prompt=self.prompt)
