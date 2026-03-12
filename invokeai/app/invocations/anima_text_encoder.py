"""Anima text encoder invocation.

Encodes text using the dual-conditioning pipeline:
1. Qwen3 0.6B: Produces hidden states (last layer)
2. T5-XXL Tokenizer: Produces token IDs only (no T5 model needed)

Both outputs are stored together in AnimaConditioningInfo and used by
the LLM Adapter inside the transformer during denoising.

Key differences from Z-Image text encoder:
- Anima uses Qwen3 0.6B (base model, NOT instruct) — no chat template
- Anima additionally tokenizes with T5-XXL tokenizer to get token IDs
- Qwen3 output includes all positions (including padding) to match ComfyUI
"""

from contextlib import ExitStack

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, T5TokenizerFast

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    AnimaConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.invocations.primitives import AnimaConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    AnimaConditioningInfo,
    ConditioningFieldData,
)

# Qwen3 max sequence length — ComfyUI's SDClipModel uses max_length=77 for Qwen3.
# We match this to ensure the LLM Adapter's cross-attention sees the same number of
# source positions (including padding) as during training.
QWEN3_MAX_SEQ_LEN = 77

# T5-XXL max sequence length for token IDs
T5_MAX_SEQ_LEN = 512

# T5-XXL tokenizer source (same vocabulary regardless of T5 model variant)
T5_TOKENIZER_NAME = "google/t5-v1_1-xxl"


@invocation(
    "anima_text_encoder",
    title="Prompt - Anima",
    tags=["prompt", "conditioning", "anima"],
    category="conditioning",
    version="1.0.1",
    classification=Classification.Prototype,
)
class AnimaTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for an Anima image.

    Uses Qwen3 0.6B for hidden state extraction and T5-XXL tokenizer for
    token IDs (no T5 model weights needed). Both are combined by the
    LLM Adapter inside the Anima transformer during denoising.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> AnimaConditioningOutput:
        qwen3_embeds, t5xxl_ids, t5xxl_weights = self._encode_prompt(context)

        # Move to CPU for storage
        qwen3_embeds = qwen3_embeds.detach().to("cpu")
        t5xxl_ids = t5xxl_ids.detach().to("cpu")
        t5xxl_weights = t5xxl_weights.detach().to("cpu") if t5xxl_weights is not None else None

        conditioning_data = ConditioningFieldData(
            conditionings=[
                AnimaConditioningInfo(
                    qwen3_embeds=qwen3_embeds,
                    t5xxl_ids=t5xxl_ids,
                    t5xxl_weights=t5xxl_weights,
                )
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return AnimaConditioningOutput(
            conditioning=AnimaConditioningField(conditioning_name=conditioning_name)
        )

    def _encode_prompt(
        self,
        context: InvocationContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode prompt using Qwen3 0.6B and T5-XXL tokenizer.

        Returns:
            Tuple of (qwen3_embeds, t5xxl_ids, t5xxl_weights).
            - qwen3_embeds: Shape (max_seq_len, 1024) — includes all positions (including padding)
              to match ComfyUI's SDClipModel behavior.
            - t5xxl_ids: Shape (seq_len,) — T5-XXL token IDs (unpadded).
            - t5xxl_weights: None (uniform weights for now).
        """
        prompt = self.prompt

        # --- Step 1: Encode with Qwen3 0.6B ---
        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            device = text_encoder.device

            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(
                    f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}."
                )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}."
                )

            context.util.signal_progress("Running Qwen3 0.6B text encoder")

            # Anima uses base Qwen3 (not instruct) — tokenize directly, no chat template
            # ComfyUI uses max_length=77 (SDClipModel default) for Qwen3
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=QWEN3_MAX_SEQ_LEN,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            if not isinstance(text_input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
                raise TypeError("Tokenizer returned unexpected types.")

            # Check for truncation
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, QWEN3_MAX_SEQ_LEN - 1 : -1])
                context.logger.warning(
                    f"Prompt truncated at {QWEN3_MAX_SEQ_LEN} tokens. Removed: {removed_text}"
                )

            # Get last hidden state from Qwen3 (ComfyUI uses layer="last")
            # Pass attention mask so padding tokens don't attend to each other,
            # but keep ALL positions in the output (including padding) to match
            # ComfyUI's SDClipModel which returns full padded sequences.
            prompt_mask = attention_mask.to(device).bool()
            outputs = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_mask,
                output_hidden_states=True,
            )

            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError("Text encoder did not return hidden_states.")
            if len(outputs.hidden_states) < 1:
                raise RuntimeError(f"Expected at least 1 hidden state, got {len(outputs.hidden_states)}.")

            # Use last hidden state — keep all positions (including padding)
            # ComfyUI's SDClipModel returns all positions without filtering.
            qwen3_embeds = outputs.hidden_states[-1][0]  # Shape: (QWEN3_MAX_SEQ_LEN, 1024)

        # --- Step 2: Tokenize with T5-XXL tokenizer (IDs only, no model) ---
        context.util.signal_progress("Tokenizing with T5-XXL")
        t5_tokenizer = T5TokenizerFast.from_pretrained(T5_TOKENIZER_NAME)
        t5_tokens = t5_tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=T5_MAX_SEQ_LEN,
            return_tensors="pt",
        )
        t5xxl_ids = t5_tokens.input_ids[0]  # Shape: (seq_len,)

        return qwen3_embeds, t5xxl_ids, None
