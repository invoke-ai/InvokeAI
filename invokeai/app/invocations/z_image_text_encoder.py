from contextlib import ExitStack
from typing import Iterator, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.invocations.primitives import ZImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_QWEN3_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ZImageConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

# Z-Image max sequence length based on diffusers default
Z_IMAGE_MAX_SEQ_LEN = 512


@invocation(
    "z_image_text_encoder",
    title="Prompt - Z-Image",
    tags=["prompt", "conditioning", "z-image"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ZImageTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a Z-Image image."""

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ZImageConditioningOutput:
        prompt_embeds = self._encode_prompt(context, max_seq_len=Z_IMAGE_MAX_SEQ_LEN)
        conditioning_data = ConditioningFieldData(conditionings=[ZImageConditioningInfo(prompt_embeds=prompt_embeds)])
        conditioning_name = context.conditioning.save(conditioning_data)
        return ZImageConditioningOutput.build(conditioning_name)

    def _encode_prompt(self, context: InvocationContext, max_seq_len: int) -> torch.Tensor:
        """Encode prompt using Qwen3 text encoder.

        Based on the ZImagePipeline._encode_prompt method from diffusers.
        """
        prompt = self.prompt
        device = TorchDevice.choose_torch_device()

        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            # Apply LoRA models to the text encoder
            lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=Z_IMAGE_LORA_QWEN3_PREFIX,
                    dtype=lora_dtype,
                )
            )

            context.util.signal_progress("Running Qwen3 text encoder")
            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(
                    f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}. "
                    "The Qwen3 encoder model may be corrupted or incompatible."
                )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}. "
                    "The Qwen3 tokenizer may be corrupted or incompatible."
                )

            # Apply chat template similar to diffusers ZImagePipeline
            # The chat template formats the prompt for the Qwen3 model
            try:
                prompt_formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except (AttributeError, TypeError) as e:
                # Fallback if tokenizer doesn't support apply_chat_template or enable_thinking
                context.logger.warning(f"Chat template failed ({e}), using raw prompt.")
                prompt_formatted = prompt

            # Tokenize the formatted prompt
            text_inputs = tokenizer(
                prompt_formatted,
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            if not isinstance(text_input_ids, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor for input_ids, got {type(text_input_ids).__name__}. "
                    "Tokenizer returned unexpected type."
                )
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor for attention_mask, got {type(attention_mask).__name__}. "
                    "Tokenizer returned unexpected type."
                )

            # Check for truncation
            untruncated_ids = tokenizer(prompt_formatted, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, max_seq_len - 1 : -1])
                context.logger.warning(
                    f"The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{max_seq_len} tokens: {removed_text}"
                )

            # Get hidden states from the text encoder
            # Use the second-to-last hidden state like diffusers does
            prompt_mask = attention_mask.to(device).bool()
            outputs = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_mask,
                output_hidden_states=True,
            )

            # Validate hidden_states output
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError(
                    "Text encoder did not return hidden_states. "
                    "Ensure output_hidden_states=True is supported by this model."
                )
            if len(outputs.hidden_states) < 2:
                raise RuntimeError(
                    f"Expected at least 2 hidden states from text encoder, got {len(outputs.hidden_states)}. "
                    "This may indicate an incompatible model or configuration."
                )
            prompt_embeds = outputs.hidden_states[-2]

            # Z-Image expects a 2D tensor [seq_len, hidden_dim] with only valid tokens
            # Based on diffusers ZImagePipeline implementation:
            # embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
            # Since batch_size=1, we take the first item and filter by mask
            prompt_embeds = prompt_embeds[0][prompt_mask[0]]

        if not isinstance(prompt_embeds, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor for prompt embeddings, got {type(prompt_embeds).__name__}. "
                "Text encoder returned unexpected type."
            )
        return prompt_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the Qwen3 text encoder."""
        for lora in self.qwen3_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
