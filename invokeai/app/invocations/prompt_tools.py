# 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Optional
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
    invocation,
    invocation_output,
)


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


@invocation(
    "pt_fields_collect", title="PTFields Collect", tags=["prompt", "fields"], category="prompt", version="1.0.0"
)
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


@invocation("pt_fields_expand", title="PTFields Expand", tags=["prompt", "fields"], category="prompt", version="1.0.0")
class PTFieldsExpandInvocation(BaseInvocation):
    """Save Expand PTFields into individual items"""

    pt_fields: str = InputField(default=None, description="PTFields in json Format", input=Input.Connection)

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
