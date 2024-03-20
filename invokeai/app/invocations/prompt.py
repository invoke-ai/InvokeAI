from os.path import exists
from typing import Optional, Union

import numpy as np
from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from pydantic import field_validator

from invokeai.app.invocations.primitives import StringCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

from .baseinvocation import BaseInvocation, invocation
from .fields import InputField, UIComponent


@invocation(
    "dynamic_prompt",
    title="Dynamic Prompt",
    tags=["prompt", "collection"],
    category="prompt",
    version="1.0.1",
    use_cache=False,
)
class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    prompt: str = InputField(
        description="The prompt to parse with dynamicprompts",
        ui_component=UIComponent.Textarea,
    )
    max_prompts: int = InputField(default=1, description="The number of prompts to generate")
    combinatorial: bool = InputField(default=False, description="Whether to use the combinatorial generator")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.combinatorial:
            generator = CombinatorialPromptGenerator()
            prompts = generator.generate(self.prompt, max_prompts=self.max_prompts)
        else:
            generator = RandomPromptGenerator()
            prompts = generator.generate(self.prompt, num_images=self.max_prompts)

        return StringCollectionOutput(collection=prompts)


@invocation(
    "prompt_from_file",
    title="Prompts from File",
    tags=["prompt", "file"],
    category="prompt",
    version="1.0.2",
)
class PromptsFromFileInvocation(BaseInvocation):
    """Loads prompts from a text file"""

    file_path: str = InputField(description="Path to prompt text file")
    pre_prompt: Optional[str] = InputField(
        default=None,
        description="String to prepend to each prompt",
        ui_component=UIComponent.Textarea,
    )
    post_prompt: Optional[str] = InputField(
        default=None,
        description="String to append to each prompt",
        ui_component=UIComponent.Textarea,
    )
    start_line: int = InputField(default=1, ge=1, description="Line in the file to start start from")
    max_prompts: int = InputField(default=1, ge=0, description="Max lines to read from file (0=all)")

    @field_validator("file_path")
    def file_path_exists(cls, v):
        if not exists(v):
            raise ValueError(FileNotFoundError)
        return v

    def promptsFromFile(
        self,
        file_path: str,
        pre_prompt: Union[str, None],
        post_prompt: Union[str, None],
        start_line: int,
        max_prompts: int,
    ):
        prompts = []
        start_line -= 1
        end_line = start_line + max_prompts
        if max_prompts <= 0:
            end_line = np.iinfo(np.int32).max
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start_line and i < end_line:
                    prompts.append((pre_prompt or "") + line.strip() + (post_prompt or ""))
                if i >= end_line:
                    break
        return prompts

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        prompts = self.promptsFromFile(
            self.file_path,
            self.pre_prompt,
            self.post_prompt,
            self.start_line,
            self.max_prompts,
        )
        return StringCollectionOutput(collection=prompts)
