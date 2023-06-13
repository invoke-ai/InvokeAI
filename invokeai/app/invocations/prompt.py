import os
from typing import Literal, Optional

from pydantic.fields import Field
from pyparsing import ParseException

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from dynamicprompts import RandomPromptGenerator, CombinatorialPromptGenerator

class PromptOutput(BaseInvocationOutput):
    """Base class for invocations that output a prompt"""
    #fmt: off
    type: Literal["prompt"] = "prompt"

    prompt: str = Field(default=None, description="The output prompt")
    #fmt: on

    class Config:
        schema_extra = {
            'required': [
                'type',
                'prompt',
            ]
        }


class PromptListOutput(BaseInvocationOutput):
    """Base class for invocations that output a list of prompts"""

    # fmt: off
    type: Literal["prompt_list"] = "prompt_list"

    prompts: list[str] = Field(description="The output prompts")
    count: int = Field(description="The size of the prompts list")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "prompts", "count"]}


class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    type: Literal["dynamic_prompt"] = "dynamic_prompt"
    prompt: str = Field(
        default=None, description="The prompt to parse with dynamicprompts"
    )
    max_prompts: int = Field(default=1, description="The number of prompts to generate")
    combinatorial: bool = Field(
        default=False, description="Whether to use the combinatorial generator"
    )

    def invoke(self, context: InvocationContext) -> PromptListOutput:
        try:
            if self.combinatorial:
                generator = CombinatorialPromptGenerator()
                prompts = generator.generate(self.prompt, max_prompts=self.max_prompts)
            else:
                generator = RandomPromptGenerator()
                prompts = generator.generate(self.prompt, num_images=self.max_prompts)
        except ParseException as e:
            warning = f"Invalid dynamic prompt: {e}"
            context.services.logger.warn(warning)
            return PromptListOutput(prompts=[self.prompt], count=1)

        return PromptListOutput(prompts=prompts, count=len(prompts))
