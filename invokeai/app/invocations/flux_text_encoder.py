from typing import Literal

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField
from invokeai.app.invocations.model import CLIPField, T5EncoderField
from invokeai.app.invocations.primitives import ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.conditioner import HFEncoder
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo


@invocation(
    "flux_text_encoder",
    title="FLUX Text Encoding",
    tags=["prompt", "conditioning", "flux"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class FluxTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a flux image."""

    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )
    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    t5_max_seq_len: Literal[256, 512] = InputField(
        description="Max sequence length for the T5 encoder. Expected to be 256 for FLUX schnell models and 512 for FLUX dev models."
    )
    prompt: str = InputField(description="Text prompt to encode.")

    # TODO(ryand): Should we create a new return type for this invocation? This ConditioningOutput is clearly not
    # compatible with other ConditioningOutputs.
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        t5_embeddings, clip_embeddings = self._encode_prompt(context)
        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=clip_embeddings, t5_embeds=t5_embeddings)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return ConditioningOutput.build(conditioning_name)

    def _encode_prompt(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor]:
        # Load CLIP.
        clip_tokenizer_info = context.models.load(self.clip.tokenizer)
        clip_text_encoder_info = context.models.load(self.clip.text_encoder)

        # Load T5.
        t5_tokenizer_info = context.models.load(self.t5_encoder.tokenizer)
        t5_text_encoder_info = context.models.load(self.t5_encoder.text_encoder)

        with (
            clip_text_encoder_info as clip_text_encoder,
            t5_text_encoder_info as t5_text_encoder,
            clip_tokenizer_info as clip_tokenizer,
            t5_tokenizer_info as t5_tokenizer,
        ):
            assert isinstance(clip_text_encoder, CLIPTextModel)
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(clip_tokenizer, CLIPTokenizer)
            assert isinstance(t5_tokenizer, T5Tokenizer)

            clip_encoder = HFEncoder(clip_text_encoder, clip_tokenizer, True, 77)
            t5_encoder = HFEncoder(t5_text_encoder, t5_tokenizer, False, self.t5_max_seq_len)

            prompt = [self.prompt]
            prompt_embeds = t5_encoder(prompt)

            pooled_prompt_embeds = clip_encoder(prompt)

        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(pooled_prompt_embeds, torch.Tensor)
        return prompt_embeds, pooled_prompt_embeds
