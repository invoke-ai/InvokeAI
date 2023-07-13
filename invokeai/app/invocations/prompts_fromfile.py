from typing import Literal, Union
from pydantic import Field

from .baseinvocation import BaseInvocation, InvocationContext
from .prompt import PromptOutput, PromptCollectionOutput

#reads prompts from a file 1 at a time then outputs them as a Prompt Collection
class PromptsFromFileInvocation(BaseInvocation):
    '''loads prompts from a file'''
    type: Literal['prompt_from_file'] = 'prompt_from_file'

    # Inputs
    filename: str = Field(default=None, description="filename of prompt file")
    pre_prompt: str = Field(default=None, description="Add to start of prompt")
    post_prompt: str = Field(default=None, description="Add to end of prompt")
    start_line: int = Field(default=0, description="line in the file start start from (0 based)")
    max_prompts: int = Field(default=999, description="Max lines to read from file")

    def invoke(self, context: InvocationContext) -> PromptCollectionOutput:
        prompts = []
        with open(self.filename) as fp:
            for i, line in enumerate(fp):

                if i >= self.start_line and i < self.start_line + self.max_prompts:
                    prompts.append(self.pre_prompt + line.strip() + self.post_prompt)
                if i >= self.start_line + self.max_prompts:
                    break

        return PromptCollectionOutput(prompt_collection=prompts, count=len(prompts))
