from typing import Optional

import torch
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
)

from invokeai.app.invocations.model import T5EncoderField
from invokeai.app.invocations.primitives import BaseInvocationOutput, FieldDescriptions, Input, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.invocation_api import (
    BaseInvocation,
    Classification,
    InputField,
    LatentsField,
    invocation,
    invocation_output,
)

from invokeai.backend.bria.bria_utils import get_t5_prompt_embeds, is_ng_none


@invocation_output("bria_text_encoder_output")
class BriaTextEncoderInvocationOutput(BaseInvocationOutput):
    """Base class for nodes that output a CogView text conditioning tensor."""

    pos_embeds: LatentsField = OutputField(description=FieldDescriptions.cond)
    neg_embeds: LatentsField = OutputField(description=FieldDescriptions.cond)
    text_ids: LatentsField = OutputField(description=FieldDescriptions.cond)


@invocation(
    "bria_text_encoder",
    title="Prompt - Bria",
    tags=["prompt", "conditioning", "bria"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaTextEncoderInvocation(BaseInvocation):
    prompt: str = InputField(
        title="Prompt",
        description="The prompt to encode",
    )
    negative_prompt: Optional[str] = InputField(
        title="Negative Prompt",
        description="The negative prompt to encode",
    )
    max_length: int = InputField(
        default=128,
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
            pos = get_t5_prompt_embeds(tokenizer, text_encoder, self.prompt, 1, self.max_length, text_encoder.device)
            neg = (
                torch.zeros_like(pos)
                if is_ng_none(self.negative_prompt)
                else get_t5_prompt_embeds(
                    tokenizer, text_encoder, self.negative_prompt, 1, self.max_length, text_encoder.device
                )
            )
            text_ids = torch.zeros((pos.shape[1], 3), device=text_encoder.device, dtype=torch.long)
        saved_pos_tensor = context.tensors.save(pos)
        saved_neg_tensor = context.tensors.save(neg)
        saved_text_ids_tensor = context.tensors.save(text_ids)
        pos_embeds_output = LatentsField(latents_name=saved_pos_tensor)
        neg_embeds_output = LatentsField(latents_name=saved_neg_tensor)
        text_ids_output = LatentsField(latents_name=saved_text_ids_tensor)
        return BriaTextEncoderInvocationOutput(
            pos_embeds=pos_embeds_output,
            neg_embeds=neg_embeds_output,
            text_ids=text_ids_output,
        )
