from typing import Literal

from pydantic.fields import Field

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from dynamicprompts.generators import RandomPromptGenerator, CombinatorialPromptGenerator

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


class PromptCollectionOutput(BaseInvocationOutput):
    """Base class for invocations that output a collection of prompts"""

    # fmt: off
    type: Literal["prompt_collection_output"] = "prompt_collection_output"

    prompt_collection: list[str] = Field(description="The output prompt collection")
    count: int = Field(description="The size of the prompt collection")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "prompt_collection", "count"]}


class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    type: Literal["dynamic_prompt"] = "dynamic_prompt"
    prompt: str = Field(description="The prompt to parse with dynamicprompts")
    max_prompts: int = Field(default=1, description="The number of prompts to generate")
    combinatorial: bool = Field(
        default=False, description="Whether to use the combinatorial generator"
    )

    def invoke(self, context: InvocationContext) -> PromptCollectionOutput:
        if self.combinatorial:
            generator = CombinatorialPromptGenerator()
            prompts = generator.generate(self.prompt, max_prompts=self.max_prompts)
        else:
            generator = RandomPromptGenerator()
            prompts = generator.generate(self.prompt, num_images=self.max_prompts)

        return PromptCollectionOutput(prompt_collection=prompts, count=len(prompts))
