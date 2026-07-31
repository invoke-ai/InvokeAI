from contextlib import ExitStack

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.invocations.primitives import Ideogram4ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.text_encoding import encode_qwen3vl_prompt
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Ideogram4ConditioningInfo,
)


@invocation(
    "ideogram4_text_encoder",
    title="Prompt - Ideogram 4",
    tags=["prompt", "conditioning", "ideogram4"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for Ideogram 4 using the Qwen3-VL encoder.

    The prompt is normally a structured JSON caption (see the Ideogram 4 prompting guide);
    plain text also works but yields lower-quality results.
    """

    prompt: str = InputField(
        description="The prompt to encode. A structured JSON caption is recommended.",
        ui_component=UIComponent.Textarea,
    )
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Ideogram4ConditioningOutput:
        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            context.util.signal_progress("Running Qwen3-VL text encoder")
            prompt_embeds = encode_qwen3vl_prompt(self.prompt, tokenizer, text_encoder)

        # Move to CPU for storage to save VRAM.
        prompt_embeds = prompt_embeds.detach().to("cpu")
        conditioning_data = ConditioningFieldData(
            conditionings=[Ideogram4ConditioningInfo(prompt_embeds=prompt_embeds)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return Ideogram4ConditioningOutput.build(conditioning_name)
