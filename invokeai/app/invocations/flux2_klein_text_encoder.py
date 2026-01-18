"""Flux2 Klein Text Encoder Invocation.

Flux2 Klein uses Qwen3 as the text encoder instead of CLIP+T5.
The key difference is that it extracts hidden states from layers [9, 18, 27]
and stacks them together for richer text representations.
"""

from contextlib import ExitStack
from typing import Iterator, Literal, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# Flux2 Klein extracts hidden states from these specific layers
# ComfyUI uses [9, 18, 27] (0-indexed layer numbers)
# hidden_states[0] is embedding layer, so layer N is at index N
KLEIN_EXTRACTION_LAYERS = [9, 18, 27]

# Default max sequence length for Klein models
KLEIN_MAX_SEQ_LEN = 512


@invocation(
    "flux2_klein_text_encoder",
    title="Prompt - Flux2 Klein",
    tags=["prompt", "conditioning", "flux", "klein", "qwen3"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2KleinTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for Flux2 Klein image generation.

    Flux2 Klein uses Qwen3 as the text encoder, extracting hidden states from
    layers [9, 18, 27] and stacking them for richer text representations.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )
    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        description="Max sequence length for the Qwen3 encoder.",
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        qwen3_embeds, pooled_embeds = self._encode_prompt(context)

        # Use FLUXConditioningInfo for compatibility with existing Flux denoiser
        # t5_embeds -> qwen3 stacked embeddings
        # clip_embeds -> pooled qwen3 embedding
        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=pooled_embeds, t5_embeds=qwen3_embeds)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _encode_prompt(self, context: InvocationContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using Qwen3 text encoder with Klein-style layer extraction.

        Returns:
            Tuple of (stacked_embeddings, pooled_embedding):
            - stacked_embeddings: Hidden states from layers [9, 18, 27] stacked together.
              Shape: (1, seq_len, hidden_size * 3)
            - pooled_embedding: Pooled representation for global conditioning.
              Shape: (1, hidden_size)
        """
        prompt = self.prompt
        device = TorchDevice.choose_torch_device()

        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (cached_weights, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            # Apply LoRA models to the text encoder
            lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_T5_PREFIX,  # Reuse T5 prefix for Qwen3 LoRAs
                    dtype=lora_dtype,
                    cached_weights=cached_weights,
                )
            )

            context.util.signal_progress("Running Qwen3 text encoder (Klein)")

            # Debug: Log model config to verify hidden_size
            if hasattr(text_encoder, "config"):
                te_config = text_encoder.config
                context.logger.info(
                    f"Qwen3 encoder config: hidden_size={getattr(te_config, 'hidden_size', 'N/A')}, "
                    f"num_hidden_layers={getattr(te_config, 'num_hidden_layers', 'N/A')}, "
                    f"model_type={getattr(te_config, 'model_type', 'N/A')}"
                )

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

            # Apply chat template matching ComfyUI's FLUX.2 Klein implementation
            # Format: <|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
            prompt_formatted = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

            # Tokenize the formatted prompt
            text_inputs = tokenizer(
                prompt_formatted,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            if not isinstance(text_input_ids, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for input_ids, got {type(text_input_ids).__name__}.")
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for attention_mask, got {type(attention_mask).__name__}.")

            # Check for truncation
            untruncated_ids = tokenizer(prompt_formatted, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, self.max_seq_len - 1 : -1])
                context.logger.warning(
                    f"The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{self.max_seq_len} tokens: {removed_text}"
                )

            # Get hidden states from specific layers [9, 18, 27]
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

            num_hidden_layers = len(outputs.hidden_states)
            context.logger.debug(f"Qwen3 encoder has {num_hidden_layers} hidden states available.")

            # Extract hidden states from Klein layers [9, 18, 27]
            # Note: hidden_states[0] is the embedding layer output, so layer N is at index N
            extracted_hidden_states = []
            for layer_idx in KLEIN_EXTRACTION_LAYERS:
                if layer_idx >= num_hidden_layers:
                    context.logger.warning(
                        f"Layer {layer_idx} not available (model has {num_hidden_layers} layers). "
                        f"Using last available layer instead."
                    )
                    layer_idx = num_hidden_layers - 1
                extracted_hidden_states.append(outputs.hidden_states[layer_idx])

            # Stack the hidden states from the 3 layers
            # Each hidden state has shape (batch_size, seq_len, hidden_size)
            # After stacking along last dim: (batch_size, seq_len, hidden_size * 3)
            stacked_embeds = torch.cat(extracted_hidden_states, dim=-1)

            # Debug: Log shapes, dtype and value ranges (no normalization - matching diffusers)
            context.logger.info(
                f"Qwen3 hidden state shapes: per_layer={extracted_hidden_states[0].shape}, "
                f"stacked={stacked_embeds.shape}, dtype={stacked_embeds.dtype}, "
                f"value_range=[{stacked_embeds.min().item():.4f}, {stacked_embeds.max().item():.4f}]"
            )

            # Create pooled embedding for global conditioning
            # Use mean pooling over the sequence (excluding padding)
            # This serves a similar role to CLIP's pooled output in standard FLUX
            last_hidden_state = outputs.hidden_states[-1]  # Use last layer for pooling
            # Expand mask to match hidden state dimensions
            expanded_mask = prompt_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            sum_embeds = (last_hidden_state * expanded_mask).sum(dim=1)
            num_tokens = expanded_mask.sum(dim=1).clamp(min=1)
            pooled_embeds = sum_embeds / num_tokens

        return stacked_embeds, pooled_embeds

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
