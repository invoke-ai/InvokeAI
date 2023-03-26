from typing import Literal

from pydantic.fields import Field

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext


class PromptOutput(BaseInvocationOutput):
    """Base class for invocations that output a prompt"""
    #fmt: off
    type: Literal["prompt"] = "prompt"

    prompt: str = Field(default=None, description="The output prompt")
    #fmt: on

class SimplePromptInvocation(BaseInvocation):
    """Simple prompt invocation."""
    #fmt: off
    type: Literal["simple_prompt"] = "simple_prompt"

    # Inputs
    prompt: str = Field(default=None, description="The prompt to output.")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(
            prompt=self.prompt
        )
