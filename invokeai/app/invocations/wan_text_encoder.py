import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import WanT5EncoderField
from invokeai.app.invocations.primitives import WanConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    WanConditioningInfo,
)

# Wan models are trained with 512-token text sequences (matches the
# upstream config.json's ``text_len: 512`` and the WanPipeline.__call__
# default). Diffusers' ``_get_t5_prompt_embeds`` has a stale 226 default
# that gets overridden by ``__call__``; using 512 here matches the actual
# pipeline behaviour.
WAN_T5_MAX_SEQ_LEN = 512


@invocation(
    "wan_text_encoder",
    title="Prompt - Wan 2.2",
    tags=["prompt", "conditioning", "wan"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanTextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Wan 2.2 using the UMT5-XXL encoder.

    Output is the encoder's last hidden state (shape: [seq_len=226, 4096]) plus
    an attention mask marking valid (non-padding) tokens. The Wan transformer
    consumes these directly as ``encoder_hidden_states``.
    """

    prompt: str = InputField(description="Text prompt for Wan 2.2.", ui_component=UIComponent.Textarea)
    wan_t5_encoder: WanT5EncoderField = InputField(
        title="UMT5-XXL Encoder",
        description=FieldDescriptions.wan_t5_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> WanConditioningOutput:
        prompt_embeds, attention_mask = self._encode(context)

        # Persist on CPU; the denoise loop will move to device as needed.
        prompt_embeds = prompt_embeds.detach().to("cpu")
        attention_mask = attention_mask.detach().to("cpu") if attention_mask is not None else None

        conditioning_data = ConditioningFieldData(
            conditionings=[
                WanConditioningInfo(prompt_embeds=prompt_embeds, prompt_attention_mask=attention_mask)
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return WanConditioningOutput.build(conditioning_name)

    def _encode(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor | None]:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from transformers import UMT5EncoderModel

        cleaned = prompt_clean(self.prompt)

        # Tokenizer + text encoder both routed through the model cache so the
        # registered loaders handle the nested-vs-flat directory layout for us
        # (main-model layout: <root>/tokenizer/ + <root>/text_encoder/;
        # standalone WanT5Encoder layout may also be flat).
        tokenizer_info = context.models.load(self.wan_t5_encoder.tokenizer)
        with tokenizer_info.model_on_device() as (_, tokenizer):
            text_inputs = tokenizer(
                [cleaned],
                padding="max_length",
                max_length=WAN_T5_MAX_SEQ_LEN,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

        text_encoder_info = context.models.load(self.wan_t5_encoder.text_encoder)
        with text_encoder_info.model_on_device() as (_, text_encoder):
            assert isinstance(text_encoder, UMT5EncoderModel)
            device = get_effective_device(text_encoder)

            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            context.util.signal_progress("Running UMT5-XXL text encoder")
            outputs = text_encoder(input_ids, attention_mask)
            # Drop the batch dim (we always encode one prompt at a time).
            prompt_embeds = outputs.last_hidden_state.squeeze(0)
            attention_mask_out = attention_mask.squeeze(0)

        # Match the Diffusers reference: zero out the embeddings past the valid
        # token count so the transformer sees clean padding.
        valid_len = int(attention_mask_out.sum().item())
        if valid_len < prompt_embeds.shape[0]:
            prompt_embeds = prompt_embeds.clone()
            prompt_embeds[valid_len:] = 0

        # If every token is valid we don't need the mask downstream.
        mask_out: torch.Tensor | None = attention_mask_out
        if attention_mask_out.all():
            mask_out = None

        return prompt_embeds.to(dtype=torch.bfloat16), mask_out
