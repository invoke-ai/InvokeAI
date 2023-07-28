from os.path import exists
from typing import Literal, Optional

import numpy as np
from pydantic import Field, validator

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext
from dynamicprompts.generators import RandomPromptGenerator, CombinatorialPromptGenerator


class PromptOutput(BaseInvocationOutput):
    """Base class for invocations that output a prompt"""

    # fmt: off
    type: Literal["prompt"] = "prompt"

    prompt: str = Field(default=None, description="The output prompt")
    # fmt: on

    class Config:
        schema_extra = {
            "required": [
                "type",
                "prompt",
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
    combinatorial: bool = Field(default=False, description="Whether to use the combinatorial generator")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Dynamic Prompt", "tags": ["prompt", "dynamic"]},
        }

    def invoke(self, context: InvocationContext) -> PromptCollectionOutput:
        if self.combinatorial:
            generator = CombinatorialPromptGenerator()
            prompts = generator.generate(self.prompt, max_prompts=self.max_prompts)
        else:
            generator = RandomPromptGenerator()
            prompts = generator.generate(self.prompt, num_images=self.max_prompts)

        return PromptCollectionOutput(prompt_collection=prompts, count=len(prompts))


class PromptsFromFileInvocation(BaseInvocation):
    """Loads prompts from a text file"""

    # fmt: off
    type: Literal['prompt_from_file'] = 'prompt_from_file'

    # Inputs
    file_path: str = Field(description="Path to prompt text file")
    pre_prompt: Optional[str] = Field(description="String to prepend to each prompt")
    post_prompt: Optional[str] = Field(description="String to append to each prompt")
    start_line: int = Field(default=1, ge=1, description="Line in the file to start start from")
    max_prompts: int = Field(default=1, ge=0, description="Max lines to read from file (0=all)")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Prompts From File", "tags": ["prompt", "file"]},
        }

    @validator("file_path")
    def file_path_exists(cls, v):
        if not exists(v):
            raise ValueError(FileNotFoundError)
        return v

    def promptsFromFile(self, file_path: str, pre_prompt: str, post_prompt: str, start_line: int, max_prompts: int):
        prompts = []
        start_line -= 1
        end_line = start_line + max_prompts
        if max_prompts <= 0:
            end_line = np.iinfo(np.int32).max
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= start_line and i < end_line:
                    prompts.append((pre_prompt or "") + line.strip() + (post_prompt or ""))
                if i >= end_line:
                    break
        return prompts

    def invoke(self, context: InvocationContext) -> PromptCollectionOutput:
        prompts = self.promptsFromFile(
            self.file_path, self.pre_prompt, self.post_prompt, self.start_line, self.max_prompts
        )
        return PromptCollectionOutput(prompt_collection=prompts, count=len(prompts))
