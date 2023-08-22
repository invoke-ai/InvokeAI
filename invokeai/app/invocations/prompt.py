from os.path import exists
from typing import Literal, Optional, Union, List

import numpy as np
import re, json

from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from pydantic import BaseModel, validator


from invokeai.app.invocations.primitives import StringCollectionOutput, StringOutput
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InputField, InvocationContext, UIComponent, UIType, tags, title

from .model import MainModelField, VAEModelField

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


@title("Dynamic Prompt")
@tags("prompt", "collection")
class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    type: Literal["dynamic_prompt"] = "dynamic_prompt"

    # Inputs
    prompt: str = InputField(description="The prompt to parse with dynamicprompts", ui_component=UIComponent.Textarea)
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


@title("Prompts from File")
@tags("prompt", "file")
class PromptsFromFileInvocation(BaseInvocation):
    """Loads prompts from a text file"""

    type: Literal["prompt_from_file"] = "prompt_from_file"

    # Inputs
    file_path: str = InputField(description="Path to prompt text file", ui_type=UIType.FilePath)
    pre_prompt: Optional[str] = InputField(
        default=None, description="String to prepend to each prompt", ui_component=UIComponent.Textarea
    )
    post_prompt: Optional[str] = InputField(
        default=None, description="String to append to each prompt", ui_component=UIComponent.Textarea
    )
    start_line: int = InputField(default=1, ge=1, description="Line in the file to start start from")
    max_prompts: int = InputField(default=1, ge=0, description="Max lines to read from file (0=all)")

    @validator("file_path")
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
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= start_line and i < end_line:
                    prompts.append((pre_prompt or "") + line.strip() + (post_prompt or ""))
                if i >= end_line:
                    break
        return prompts

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        prompts = self.promptsFromFile(
            self.file_path, self.pre_prompt, self.post_prompt, self.start_line, self.max_prompts
        )
        return StringCollectionOutput(prompt_collection=prompts)


class PromptsToFileInvocationOutput(BaseInvocationOutput):
    """Base class for invocation that writes to a file and returns nothing of use"""
    #fmt: off
    type: Literal["prompt_to_file_output"] = "prompt_to_file_output"
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
    prompts: Union[str, list[str], None] = Field(default=None, description="Prompt or collection of prompts to write")
    append: bool = Field(default=True, description="Append or overwrite file")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Prompts To File",
                "tags": ["prompt", "file"],
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

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Prompt Spilt Negative",
                "tags": ["prompt", "split", "negative"]
            },
        }

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

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Prompt Join",
                "tags": ["prompt", "join"]
            },
        }

    def invoke(self, context: InvocationContext) -> PromptOutput:
        return PromptOutput(prompt=((self.prompt_a or '') + (self.prompt_b or '')))  


class PromptReplaceInvocation(BaseInvocation):
    """Replaces the search string with the replace string in the prompt"""

    type: Literal["prompt_replace"] = "prompt_replace"
    prompt: str = Field(default='', description="Prompt to work on")
    search_string : str = Field(default='', description="String to search for")
    replace_string : str = Field(default='', description="String to replace the search")
    use_regex: bool = Field(default=False, description="Use search string as a regex expression (non regex is case insensitive)")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Prompt Replace",
                "tags": ["prompt", "replace", "regex"]
            },
        }

    def invoke(self, context: InvocationContext) -> PromptOutput:
        pattern = (self.search_string or '')
        new_prompt = (self.prompt or '')
        if len(pattern) > 0: 
            if not self.use_regex:
                #None regex so make case insensitve 
                pattern = "(?i)" + re.escape(pattern)
            new_prompt = re.sub(pattern, (self.replace_string or ''), new_prompt)
        return PromptOutput(prompt=new_prompt)  
    
    class PTFields(BaseModel):
    """Prompt Tools Fields for an image generated in InvokeAI."""
    positive_prompt: str = Field(default='', description="The positive prompt parameter")
    positive_style_prompt: str = Field(default='', description="The positive style prompt parameter")
    negative_prompt: str = Field(default='', description="The negative prompt parameter")
    negative_style_prompt: str = Field(default='', description="The negative prompt parameter")
    width: int = Field(default=512, description="The width parameter")
    height: int = Field(default=512, description="The height parameter")
    seed: int = Field(default=0, description="The seed used for noise generation")
    cfg_scale: float = Field(default=7.0, description="The classifier-free guidance scale parameter")
    steps: int = Field(default=10, description="The number of steps used for inference")
#     class Config:
#         schema_extra = {
#             "required": ["positive_prompt", "positive_style_prompt", "negative_prompt"],
#             "ui": {
#                 "type_hints": {
# #                    "model": "model",
# #                    "vae": "vae_model",
#                     },
#                 },
#             }

class PTFieldsCollectOutput(BaseInvocationOutput):
    """PTFieldsCollect Output"""
    type: Literal["pt_fields_collect_output"] = "pt_fields_collect_output"

    pt_fields: str = Field(description="PTFields in Json Format")

class PTFieldsCollectInvocation(BaseInvocation):
    """Prompt Tools Fields for an image generated in InvokeAI."""
    type: Literal["pt_fields_collect"] = "pt_fields_collect"

    positive_prompt: str = Field(default='', description="The positive prompt parameter")
    positive_style_prompt: str = Field(default='', description="The positive style prompt parameter")
    negative_prompt: str = Field(default='', description="The negative prompt parameter")
    negative_style_prompt: str = Field(default='', description="The negative prompt parameter")
    width: int = Field(default=512, description="The width parameter")
    height: int = Field(default=512, description="The height parameter")
    seed: int = Field(default=0, description="The seed used for noise generation")
    cfg_scale: float = Field(default=7.0, description="The classifier-free guidance scale parameter")
    steps: int = Field(default=10, description="The number of steps used for inference")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "PTFields Collect",
                "tags": ["prompt", "file"],
            },
        }
       
    def invoke(self, context: InvocationContext) -> PTFieldsCollectOutput:
        x:str = str(json.dumps(
                    PTFields(
                        positive_prompt = self.positive_prompt, 
                        positive_style_prompt = self.positive_style_prompt,
                        negative_prompt = self.negative_prompt,
                        negative_style_prompt = self.negative_style_prompt,
                        width = self.width,
                        height = self.height,
                        seed = self.seed,
                        cfg_scale = self.cfg_scale,
                        steps = self.steps,
                ).dict()
            )
        )
        return PTFieldsCollectOutput(pt_fields=x)


class PTFieldsExpandOutput(BaseInvocationOutput):
    """Prompt Tools Fields for an image generated in InvokeAI."""
    type: Literal["pt_fields_expand_output"] = "pt_fields_expand_output"    

    positive_prompt: str = Field(description="The positive prompt parameter")
    positive_style_prompt: str = Field(description="The positive style prompt parameter")
    negative_prompt: str = Field(description="The negative prompt parameter")
    negative_style_prompt: str = Field(description="The negative prompt parameter")
    width: int = Field(description="The width parameter")
    height: int = Field(description="The height parameter")
    seed: int = Field(description="The seed used for noise generation")
    cfg_scale: float = Field(description="The classifier-free guidance scale parameter")
    steps: int = Field(description="The number of steps used for inference")

    class Config:
        schema_extra = {
            'required': ['type'],
            "ui": {
                "type_hints": {
                    },
                },
            }
        
class PTFieldsExpandInvocation(BaseInvocation):
    '''Save Expand PTFields into individual items'''
    type: Literal['pt_fields_expand'] = 'pt_fields_expand'
    pt_fields: str = Field(default=None, description="PTFields in Json Format")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "PTFields Expand",
                "tags": ["prompt", "file"],
                "type_hints": {
                    "pt_fields": "string",
                }
            },
        }
       
    def invoke(self, context: InvocationContext) -> PTFieldsExpandOutput:
        fields = json.loads(self.pt_fields)

        return PTFieldsExpandOutput(
            positive_prompt = fields.get('positive_prompt'),
            positive_style_prompt = fields.get('positive_style_prompt'),
            negative_prompt = fields.get('negative_prompt'),
            negative_style_prompt = fields.get('negative_style_prompt'),
            width = fields.get('width'),
            height = fields.get('height'),
            seed = fields.get('seed'),
            cfg_scale = fields.get('cfg_scale'),
            steps = fields.get('steps'),
        )
