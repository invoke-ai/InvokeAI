# 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Literal, Union, Optional
import re
import json

from pydantic import BaseModel

from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    Input,
    OutputField,
    InvocationContext,
    UIComponent,
    UIType,
    invocation,
    invocation_output,
)


@invocation_output("prompt_to_file_output")
class PromptsToFileInvocationOutput(BaseInvocationOutput):
    """Base class for invocation that writes to a file and returns nothing of use"""


@invocation("prompt_to_file", title="Prompts To File", tags=["prompt", "file"], category="prompt")
class PromptsToFileInvocation(BaseInvocation):
    """Save prompts to a text file"""

    file_path: str = InputField(description="Path to prompt text file")
    prompts: Union[str, list[str], None] = InputField(
        default=None, description="Prompt or collection of prompts to write", ui_type=UIType.Collection
    )
    append: bool = InputField(default=True, description="Append or overwrite file")

    def invoke(self, context: InvocationContext) -> PromptsToFileInvocationOutput:
        with open(self.file_path, "a" if self.append else "w") as f:
            if isinstance(self.prompts, list):
                for line in self.prompts:
                    f.write(line + "\n")
            else:
                f.write((self.prompts or "") + "\n")

        return PromptsToFileInvocationOutput()


@invocation_output("prompt_pos_neg_output")
class PromptPosNegOutput(BaseInvocationOutput):
    """Base class for invocations that output a posirtive and negative prompt"""

    positive_prompt: str = OutputField(description="Positive prompt")
    negative_prompt: str = OutputField(description="Negative prompt")


@invocation("prompt_split_neg", title="Prompt Split Negative", tags=["prompt", "split", "negative"], category="prompt")
class PromptSplitNegInvocation(BaseInvocation):
    """Splits prompt into two prompts, inside [] goes into negative prompt everthing else goes into positive prompt. Each [ and ] character is replaced with a space"""

    prompt: str = InputField(default="", description="Prompt to split")

    def invoke(self, context: InvocationContext) -> PromptPosNegOutput:
        p_prompt = ""
        n_prompt = ""
        brackets_depth = 0
        escaped = False

        for char in self.prompt or "":
            if char == "[" and not escaped:
                n_prompt += " "
                brackets_depth += 1
            elif char == "]" and not escaped:
                brackets_depth -= 1
                char = " "
            elif brackets_depth > 0:
                n_prompt += char
            else:
                p_prompt += char

            # keep track of the escape char but only if it isn't escaped already
            if char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False

        return PromptPosNegOutput(positive_prompt=p_prompt, negative_prompt=n_prompt)


@invocation("prompt_join", title="Prompt Join", tags=["prompt", "join"], category="prompt")
class PromptJoinInvocation(BaseInvocation):
    """Joins prompt left to prompt right"""

    prompt_left: str = InputField(default="", description="Prompt Left", ui_component=UIComponent.Textarea)
    prompt_right: str = InputField(default="", description="Prompt Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.prompt_left or "") + (self.prompt_right or "")))


@invocation("prompt_join_three", title="Prompt Join Three", tags=["prompt", "join"], category="prompt")
class PromptJoinThreeInvocation(BaseInvocation):
    """Joins prompt left to prompt middle to prompt right"""

    prompt_left: str = InputField(default="", description="Prompt Left", ui_component=UIComponent.Textarea)
    prompt_middle: str = InputField(default="", description="Prompt Middle", ui_component=UIComponent.Textarea)
    prompt_right: str = InputField(default="", description="Prompt Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.prompt_left or "") + (self.prompt_middle or "") + (self.prompt_right or "")))


@invocation("prompt_replace", title="Prompt Replace", tags=["prompt", "replace", "regex"], category="prompt")
class PromptReplaceInvocation(BaseInvocation):
    """Replaces the search string with the replace string in the prompt"""

    prompt: str = InputField(default="", description="Prompt to work on", ui_component=UIComponent.Textarea)
    search_string: str = InputField(default="", description="String to search for", ui_component=UIComponent.Textarea)
    replace_string: str = InputField(
        default="", description="String to replace the search", ui_component=UIComponent.Textarea
    )
    use_regex: bool = InputField(
        default=False, description="Use search string as a regex expression (non regex is case insensitive)"
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        pattern = self.search_string or ""
        new_prompt = self.prompt or ""
        if len(pattern) > 0:
            if not self.use_regex:
                # None regex so make case insensitve
                pattern = "(?i)" + re.escape(pattern)
            new_prompt = re.sub(pattern, (self.replace_string or ""), new_prompt)
        return StringOutput(value=new_prompt)


class PTFields(BaseModel):
    """Prompt Tools Fields for an image generated in InvokeAI."""

    positive_prompt: str
    positive_style_prompt: str
    negative_prompt: str
    negative_style_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    cfg_scale: float


@invocation_output("pt_fields_collect_output")
class PTFieldsCollectOutput(BaseInvocationOutput):
    """PTFieldsCollect Output"""

    pt_fields: str = OutputField(description="PTFields in Json Format")


@invocation("pt_fields_collect", title="PTFields Collect", tags=["prompt", "fields"], category="prompt")
class PTFieldsCollectInvocation(BaseInvocation):
    """Collect Prompt Tools Fields for an image generated in InvokeAI."""

    positive_prompt: Optional[str] = InputField(description="The positive prompt parameter", input=Input.Connection)
    positive_style_prompt: Optional[str] = InputField(
        description="The positive style prompt parameter", input=Input.Connection
    )
    negative_prompt: Optional[str] = InputField(description="The negative prompt parameter", input=Input.Connection)
    negative_style_prompt: Optional[str] = InputField(
        description="The negative prompt parameter", input=Input.Connection
    )
    seed: Optional[int] = InputField(description="The seed used for noise generation", input=Input.Connection)
    width: Optional[int] = InputField(description="The width parameter", input=Input.Connection)
    height: Optional[int] = InputField(description="The height parameter", input=Input.Connection)
    steps: Optional[int] = InputField(description="The number of steps used for inference", input=Input.Connection)
    cfg_scale: Optional[float] = InputField(
        description="The classifier-free guidance scale parameter", input=Input.Connection
    )

    def invoke(self, context: InvocationContext) -> PTFieldsCollectOutput:
        x: str = str(
            json.dumps(
                PTFields(
                    positive_prompt=self.positive_prompt,
                    positive_style_prompt=self.positive_style_prompt,
                    negative_prompt=self.negative_prompt,
                    negative_style_prompt=self.negative_style_prompt,
                    seed=self.seed,
                    width=self.width,
                    height=self.height,
                    steps=self.steps,
                    cfg_scale=self.cfg_scale,
                ).dict()
            )
        )
        return PTFieldsCollectOutput(pt_fields=x)


@invocation_output("pt_fields_expand_output")
class PTFieldsExpandOutput(BaseInvocationOutput):
    """Expand Prompt Tools Fields for an image generated in InvokeAI."""

    positive_prompt: str = OutputField(description="The positive prompt parameter")
    positive_style_prompt: str = OutputField(description="The positive style prompt parameter")
    negative_prompt: str = OutputField(description="The negative prompt parameter")
    negative_style_prompt: str = OutputField(description="The negative prompt parameter")
    seed: int = OutputField(description="The seed used for noise generation")
    width: int = OutputField(description="The width parameter")
    height: int = OutputField(description="The height parameter")
    steps: int = OutputField(description="The number of steps used for inference")
    cfg_scale: float = OutputField(description="The classifier-free guidance scale parameter")


@invocation("pt_fields_expand", title="PTFields Expand", tags=["prompt", "fields"], category="prompt")
class PTFieldsExpandInvocation(BaseInvocation):
    """Save Expand PTFields into individual items"""

    pt_fields: str = InputField(default=None, description="PTFields in json Format", input=Input.Connection)

    def invoke(self, context: InvocationContext) -> PTFieldsExpandOutput:
        fields = json.loads(self.pt_fields)

        return PTFieldsExpandOutput(
            positive_prompt=fields.get("positive_prompt"),
            positive_style_prompt=fields.get("positive_style_prompt"),
            negative_prompt=fields.get("negative_prompt"),
            negative_style_prompt=fields.get("negative_style_prompt"),
            width=fields.get("width"),
            height=fields.get("height"),
            seed=fields.get("seed"),
            cfg_scale=fields.get("cfg_scale"),
            steps=fields.get("steps"),
        )


@invocation("prompt_strength", title="Prompt Strength", tags=["prompt"], category="prompt")
class PromptStrengthInvocation(BaseInvocation):
    """Takes a prompt string and float strength and outputs a new string in the format of (prompt)strength"""

    prompt: str = InputField(default="", description="Prompt to work on", ui_component=UIComponent.Textarea)
    strength: float = InputField(default=1, gt=0, description="strength of the prompt")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=f"({self.prompt}){self.strength}")


COMBINE_TYPE = Literal[".and", ".blend"]


@invocation("prompt_strengths_combine", title="Prompt Strengths Combine", tags=["prompt", "combine"], category="prompt")
class PromptStrengthsCombineInvocation(BaseInvocation):
    """Takes a collection of prompt strength strings and converts it into a combined .and() or .blend() structure. Blank prompts are ignored"""

    prompt_strengths: list[str] = InputField(
        default=[""], description="Prompt strengths to combine", ui_type=UIType.Collection
    )
    combine_type: COMBINE_TYPE = InputField(
        default=".and", description="Combine type .and() or .blend()", input=Input.Direct
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        strings = []
        numbers = []
        for item in self.prompt_strengths:
            string, number = item.rsplit(")", 1)
            string = string[1:].strip()
            number = float(number)
            if len(string) > 0:
                strings.append(f'"{string}"')
                numbers.append(number)
        return StringOutput(value=f'({",".join(strings)}){self.combine_type}({",".join(map(str, numbers))})')
