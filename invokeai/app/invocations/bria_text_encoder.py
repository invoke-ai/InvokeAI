from typing import Literal, Optional

import torch
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
)

from invokeai.app.invocations.model import T5EncoderField
from invokeai.app.invocations.primitives import BaseInvocationOutput, FieldDescriptions, Input, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.bria.pipeline_bria_controlnet import encode_prompt
from invokeai.invocation_api import (
    BaseInvocation,
    Classification,
    InputField,
    FluxConditioningField,
    invocation,
    invocation_output,
)


@invocation_output("bria_text_encoder_output")
class BriaTextEncoderInvocationOutput(BaseInvocationOutput):
    """Base class for nodes that output a Bria text conditioning tensor."""

    pos_embeds: FluxConditioningField = OutputField(description=FieldDescriptions.cond)
    neg_embeds: FluxConditioningField = OutputField(description=FieldDescriptions.cond)


@invocation(
    "bria_text_encoder",
    title="Prompt - Bria",
    tags=["prompt", "conditioning", "bria"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaTextEncoderInvocation(BaseInvocation):
    """
    Encode a prompt into a Bria text conditioning tensor.
    """

    prompt: str = InputField(
        title="Prompt",
        description="The prompt to encode",
    )
    negative_prompt: Optional[str] = InputField(
        title="Negative Prompt",
        description="The negative prompt to encode",
        default="Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate",
    )
    max_length: int = InputField(
        default=256,
        ge=128,
        le=512,
        title="Max Length",
        description="The maximum length of the prompt",
    )
    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> BriaTextEncoderInvocationOutput:
        t5_encoder_info = context.models.load(self.t5_encoder.text_encoder)
        t5_tokenizer_info = context.models.load(self.t5_encoder.tokenizer)
        with (
            t5_encoder_info as text_encoder,
            t5_tokenizer_info as tokenizer,
        ):
            assert isinstance(tokenizer, T5TokenizerFast)
            assert isinstance(text_encoder, T5EncoderModel)

        prompt_embeds, negative_prompt_embeds = encode_prompt(
            prompt=self.prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            negative_prompt=self.negative_prompt,
            device=text_encoder.device,
            num_images_per_prompt=1,
            max_sequence_length=self.max_length,
            lora_scale=1.0,
        )

        saved_pos_tensor = context.tensors.save(prompt_embeds)
        saved_neg_tensor = context.tensors.save(negative_prompt_embeds)
        pos_embeds_output = FluxConditioningField(conditioning_name=saved_pos_tensor)
        neg_embeds_output = FluxConditioningField(conditioning_name=saved_neg_tensor)
        return BriaTextEncoderInvocationOutput(
            pos_embeds=pos_embeds_output,
            neg_embeds=neg_embeds_output,
        )
