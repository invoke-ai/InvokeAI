from os.path import exists
from typing import Literal, Optional, Union, List

import numpy as np
import re
from pydantic import Field, validator

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig
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
    

class PromptsFromFileInvocation(BaseInvocation):
    '''Loads prompts from a text file'''
    # fmt: off
    type: Literal['prompt_from_file'] = 'prompt_from_file'

    # Inputs
    file_path: str = Field(description="Path to prompt text file")
    pre_prompt: Optional[str] = Field(description="String to prepend to each prompt")
    post_prompt: Optional[str] = Field(description="String to append to each prompt")
    start_line: int = Field(default=1, ge=1, description="Line in the file to start start from")
    max_prompts: int = Field(default=1, ge=0, description="Max lines to read from file (0=all)")
    #fmt: on

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
                    prompts.append((pre_prompt or '') + line.strip() + (post_prompt or ''))
                if i >= end_line:
                    break
        return prompts

    def invoke(self, context: InvocationContext) -> PromptCollectionOutput:
        prompts = self.promptsFromFile(self.file_path, self.pre_prompt, self.post_prompt, self.start_line, self.max_prompts)
        return PromptCollectionOutput(prompt_collection=prompts, count=len(prompts))


class PromptsToFileInvocationOutput(BaseInvocationOutput):
    """Base class for invocation that writes to a file and returns nothing of use"""
    #fmt: off
    type: Literal["prompts_to_file_output"] = "prompts_to_file_output"
    #fmt: on

    class Config:
        schema_extra = {
            'required': [
                'type'
            ]
        }

class PromptsToFileInvocation(BaseInvocation):
    '''Save prompts to a text file'''
    # fmt: off
    type: Literal['prompt_to_file'] = 'prompt_to_file'

    # Inputs
    file_path: str = Field(description="Path to prompt text file")
    prompts: Union[str, list[str], None] = Field(default=None, description="Collection of prompts to write")
    append: bool = Field(default=True, description="Append or overwrite file")
    #fmt: on


    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "type_hints": {
                    "prompts": "string",
                }
            },
        }
        
    def invoke(self, context: InvocationContext) -> PromptsToFileInvocationOutput:
        if self.append:
            file_mode = 'a'
        else:
            file_mode = 'w'

        with open(self.file_path, file_mode) as f:
            if isinstance(self.prompts, list):
                for line in (self.prompts):
                    f.write ( line + '\n' )
            else:
                f.write((self.prompts or '') + '\n')
 
        return PromptsToFileInvocationOutput()


class PromptPosNegOutput(BaseInvocationOutput):
    """Base class for invocations that output a posirtive and negative prompt"""

    # fmt: off
    type: Literal["prompt_pos_neg_output"] = "prompt_pos_neg_output"

    positive_prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(description="Negative prompt")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "positive_prompt", "negative_prompt"]}


class PromptSplitNegInvocation(BaseInvocation):
    """Splits prompt into two prompts, inside [] goes into negative prompt everthing else goes into positive prompt. Each [ and ] character is replaced with a space"""

    type: Literal["prompt_split_neg"] = "prompt_split_neg"
    prompt: str = Field(default='', description="Prompt to split")

    def invoke(self, context: InvocationContext) -> PromptPosNegOutput:
        p_prompt = ""
        n_prompt = ""
        brackets_depth = 0
        escaped = False

        for char in (self.prompt or ''):
            if char == "[" and not escaped:
                n_prompt += ' '
                brackets_depth += 1 
            elif char == "]" and not escaped:
                brackets_depth -= 1 
                char = ' ' 
            elif brackets_depth > 0:
                n_prompt += char
            else:
                p_prompt += char            

            #keep track of the escape char but only if it isn't escaped already
            if char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False

        return PromptPosNegOutput(positive_prompt=p_prompt, negative_prompt=n_prompt)


class PromptJoinInvocation(BaseInvocation):
    """Joins prompt a to prompt b"""

    type: Literal["prompt_join"] = "prompt_join"
    prompt_a: str = Field(default='', description="Prompt a - (Left)")
    prompt_b: str = Field(default='', description="Prompt b - (Right)")

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(prompt=((self.prompt_a or '') + (self.prompt_b or '')))  


class PromptReplaceInvocation(BaseInvocation):
    """Replaces the search string with the replace string in the prompt"""

    type: Literal["prompt_replace"] = "prompt_replace"
    prompt: str = Field(default='', description="Prompt to work on")
    search_string : str = Field(default='', description="String to search for")
    replace_string : str = Field(default='', description="String to replace the search")
    use_regex: bool = Field(default=False, description="Use search string as a regex expression (non regex is case insensitive)")

    def invoke(self, context: InvocationContext) -> PromptOutput:
        pattern = (self.search_string or '')
        new_prompt = (self.prompt or '')
        if len(pattern) > 0: 
            if not self.use_regex:
                #None regex so make case insensitve 
                pattern = "(?i)" + re.escape(pattern)
            new_prompt = re.sub(pattern, (self.replace_string or ''), new_prompt)
        return PromptOutput(prompt=new_prompt)  
