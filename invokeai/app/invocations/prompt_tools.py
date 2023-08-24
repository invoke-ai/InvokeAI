from typing import Literal, Union
import re
import json

from pydantic import BaseModel

from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, InputField, OutputField, InvocationContext, UIComponent, UIType, tags, title


class PromptsToFileInvocationOutput(BaseInvocationOutput):
    """Base class for invocation that writes to a file and returns nothing of use"""

    type: Literal["prompt_to_file_output"] = "prompt_to_file_output"

@title("Prompts To File")
@tags("prompt", "file")
class PromptsToFileInvocation(BaseInvocation):
    '''Save prompts to a text file'''

    type: Literal['prompt_to_file'] = 'prompt_to_file'

    # Inputs - Prompts should allow str and list(str) but only collection until fix available
    file_path: str = InputField(description="Path to prompt text file")
    prompts: Union[str, list[str], None] = InputField(default=None, description="Prompt or collection of prompts to write", ui_type=UIType.Collection)
    append: bool = InputField(default=True, description="Append or overwrite file")
        
    def invoke(self, context: InvocationContext) -> PromptsToFileInvocationOutput:
        with open(self.file_path, 'a' if self.append else 'w') as f:
            if isinstance(self.prompts, list):
                for line in (self.prompts):
                    f.write ( line + '\n' )
            else:
                f.write((self.prompts or '') + '\n')
 
        return PromptsToFileInvocationOutput()


class PromptPosNegOutput(BaseInvocationOutput):
    """Base class for invocations that output a posirtive and negative prompt"""

    type: Literal["prompt_pos_neg_output"] = "prompt_pos_neg_output"

    # Outputs
    positive_prompt: str = OutputField(description="Positive prompt")
    negative_prompt: str = OutputField(description="Negative prompt")

@title("Prompt Spilt Negative")
@tags("prompt", "split", "negative")
class PromptSplitNegInvocation(BaseInvocation):
    """Splits prompt into two prompts, inside [] goes into negative prompt everthing else goes into positive prompt. Each [ and ] character is replaced with a space"""

    type: Literal["prompt_split_neg"] = "prompt_split_neg"

    # Inputs
    prompt: str = InputField(default='', description="Prompt to split")

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
    

@title("Prompt Join")
@tags("prompt", "join")
class PromptJoinInvocation(BaseInvocation):
    """Joins prompt left to prompt right"""

    type: Literal["prompt_join"] = "prompt_join"

    # Inputs
    prompt_left: str = InputField(default='', description="Prompt Left", ui_component=UIComponent.Textarea)
    prompt_right: str = InputField(default='', description="Prompt Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.prompt_left or '') + (self.prompt_right or '')))  


@title("Prompt Join Three")
@tags("prompt", "join")
class PromptJoinThreeInvocation(BaseInvocation):
    """Joins prompt left to prompt middle to prompt right"""

    type: Literal["prompt_join_three"] = "prompt_join_three"

    # Inputs
    prompt_left: str = InputField(default='', description="Prompt Left", ui_component=UIComponent.Textarea)
    prompt_middle: str = InputField(default='', description="Prompt Middle)", ui_component=UIComponent.Textarea)
    prompt_right: str = InputField(default='', description="Prompt Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.prompt_left or '') + (self.prompt_middle or '') + (self.prompt_right or '')))  


@title("Prompt Replace")
@tags("prompt", "replace", "regex")
class PromptReplaceInvocation(BaseInvocation):
    """Replaces the search string with the replace string in the prompt"""

    type: Literal["prompt_replace"] = "prompt_replace"

    # Inputs
    prompt: str = InputField(default='', description="Prompt to work on", ui_component=UIComponent.Textarea)
    search_string : str = InputField(default='', description="String to search for", ui_component=UIComponent.Textarea)
    replace_string : str = InputField(default='', description="String to replace the search", ui_component=UIComponent.Textarea)
    use_regex: bool = InputField(default=False, description="Use search string as a regex expression (non regex is case insensitive)")

    def invoke(self, context: InvocationContext) -> StringOutput:
        pattern = (self.search_string or '')
        new_prompt = (self.prompt or '')
        if len(pattern) > 0: 
            if not self.use_regex:
                #None regex so make case insensitve 
                pattern = "(?i)" + re.escape(pattern)
            new_prompt = re.sub(pattern, (self.replace_string or ''), new_prompt)
        return StringOutput(value=new_prompt)  


class PTFields(BaseModel):
    """Prompt Tools Fields for an image generated in InvokeAI."""
    positive_prompt: str
    positive_style_prompt: str
    negative_prompt: str
    negative_style_prompt: str
    width: int
    height: int
    seed: int
    cfg_scale: float
    steps: int

class PTFieldsCollectOutput(BaseInvocationOutput):
    """PTFieldsCollect Output"""
    type: Literal["pt_fields_collect_output"] = "pt_fields_collect_output"

    # Outputs
    pt_fields: str = OutputField(description="PTFields in Json Format")

@title("PTFields Collect")
@tags("prompt", "file")
class PTFieldsCollectInvocation(BaseInvocation):
    """Prompt Tools Fields for an image generated in InvokeAI."""
    type: Literal["pt_fields_collect"] = "pt_fields_collect"

    # Inputs
    positive_prompt: str = InputField(default='', description="The positive prompt parameter")
    positive_style_prompt: str = InputField(default='', description="The positive style prompt parameter")
    negative_prompt: str = InputField(default='', description="The negative prompt parameter")
    negative_style_prompt: str = InputField(default='', description="The negative prompt parameter")
    width: int = InputField(default=512, description="The width parameter")
    height: int = InputField(default=512, description="The height parameter")
    seed: int = InputField(default=0, description="The seed used for noise generation")
    cfg_scale: float = InputField(default=7.0, description="The classifier-free guidance scale parameter")
    steps: int = InputField(default=10, description="The number of steps used for inference")
       
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

    # Outputs
    positive_prompt: str = OutputField(description="The positive prompt parameter")
    positive_style_prompt: str = OutputField(description="The positive style prompt parameter")
    negative_prompt: str = OutputField(description="The negative prompt parameter")
    negative_style_prompt: str = OutputField(description="The negative prompt parameter")
    width: int = OutputField(description="The width parameter")
    height: int = OutputField(description="The height parameter")
    seed: int = OutputField(description="The seed used for noise generation")
    cfg_scale: float = OutputField(description="The classifier-free guidance scale parameter")
    steps: int = OutputField(description="The number of steps used for inference")
        
@title("PTFields Expand")
@tags("prompt", "file")
class PTFieldsExpandInvocation(BaseInvocation):
    '''Save Expand PTFields into individual items'''
    type: Literal['pt_fields_expand'] = 'pt_fields_expand'

    # Inputs
    pt_fields: str = InputField(default=None, description="PTFields in json Format")
       
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
