"""Anima text encoder invocation.

Encodes text using the dual-conditioning pipeline:
1. Qwen3 0.6B: Produces hidden states (last layer)
2. T5-XXL Tokenizer: Produces token IDs only (no T5 model needed)

Both outputs are stored together in AnimaConditioningInfo and used by
the LLM Adapter inside the transformer during denoising.

Anima-3.8B additionally reads Qwen3.5 4B hidden states (layers 7, 15, 23, 31) through the semantic
connector bundled in its transformer. They are encoded here when a Qwen3.5 encoder is connected and
stored beside the Qwen3 ones; the connector itself runs inside the denoising loop, per step.

Key differences from Z-Image text encoder:
- Anima uses Qwen3 0.6B (base model, NOT instruct) — no chat template
- Anima additionally tokenizes with T5-XXL tokenizer to get token IDs
- Qwen3 output uses all positions (including padding) for full context
"""

from contextlib import ExitStack
from typing import Iterator

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    AnimaConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3EncoderField, Qwen35EncoderField
from invokeai.app.invocations.primitives import AnimaConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.anima.semantic_connector import QWEN35_LAYER_INDICES
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.anima_lora_constants import ANIMA_LORA_QWEN3_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    AnimaConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.t5.t5_tokenizer import load_bundled_t5_tokenizer
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

# T5-XXL max sequence length for token IDs
T5_MAX_SEQ_LEN = 512

# Safety cap for Qwen3 sequence length to prevent GPU OOM on extremely long prompts.
# Qwen3 0.6B supports 32K context but the LLM Adapter doesn't need that much.
QWEN3_MAX_SEQ_LEN = 8192

# The reference encoder for Anima-3.8B's semantic connector truncates the Qwen3.5 prompt here.
QWEN3_5_MAX_SEQ_LEN = 1024


@invocation(
    "anima_text_encoder",
    title="Prompt - Anima",
    tags=["prompt", "conditioning", "anima"],
    category="conditioning",
    version="1.5.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class AnimaTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for an Anima image.

    Uses Qwen3 0.6B for hidden state extraction and a bundled T5-XXL tokenizer for
    token IDs (no T5 model weights needed). Both are combined by the
    LLM Adapter inside the Anima transformer during denoising.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )
    mask: TensorField | None = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )
    qwen3_5_encoder: Qwen35EncoderField | None = InputField(
        default=None,
        title="Qwen3.5 Encoder",
        description=f"{FieldDescriptions.qwen3_5_encoder}. Connect it for Anima-3.8B, which reads both encoders.",
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> AnimaConditioningOutput:
        qwen3_embeds, t5xxl_ids, t5xxl_weights = self._encode_prompt(context)
        qwen35_states, qwen35_mask = self._encode_qwen3_5(context) if self.qwen3_5_encoder else (None, None)

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
                    qwen35_states=qwen35_states,
                    qwen35_mask=qwen35_mask,
                )
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return AnimaConditioningOutput(
            conditioning=AnimaConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _encode_prompt(
        self,
        context: InvocationContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode prompt using Qwen3 0.6B and T5-XXL tokenizer.

        Returns:
            Tuple of (qwen3_embeds, t5xxl_ids, t5xxl_weights).
            - qwen3_embeds: Shape (max_seq_len, 1024) — includes all positions (including padding)
              to preserve full sequence context for the LLM Adapter.
            - t5xxl_ids: Shape (seq_len,) — T5-XXL token IDs (unpadded).
            - t5xxl_weights: None (uniform weights for now).
        """
        prompt = self.prompt

        # --- Step 1: Encode with Qwen3 0.6B ---
        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)
        text_encoder_format = context.models.get_config(self.qwen3_encoder.text_encoder).format
        # An nvfp4 or GGUF build dequantizes each packed Linear per forward, a transient its resident size does not
        # cover. Read from the unlocked model, before the lock the reservation applies to; zero for other builds.
        dequant_bytes = peak_dequant_transient_bytes(
            text_encoder_info.model, TorchDevice.choose_anima_inference_dtype(text_encoder_info.compute_device)
        )

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(
                text_encoder_info.model_on_device(working_mem_bytes=dequant_bytes)
            )
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            # Use the encoder's intended compute device, not its current parameter residency: partial loading may
            # have temporarily offloaded all weights to RAM, which would wrongly run the whole encode on the CPU (see
            # #9373). Qwen3 is fully autocast-capable, so nothing pins it to the compute device otherwise.
            device = text_encoder_info.compute_device

            # Apply LoRA models to the text encoder
            lora_dtype = TorchDevice.choose_anima_inference_dtype(device)
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=ANIMA_LORA_QWEN3_PREFIX,
                    dtype=lora_dtype,
                    # Quantized weights (packed nvfp4 Linears, GGUF, SDNQ) cannot take a direct patch, so
                    # their LoRAs ride as sidecars.
                    force_sidecar_patching=requires_sidecar_patching(text_encoder, text_encoder_format),
                )
            )

            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}.")
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}.")

            context.util.signal_progress("Running Qwen3 0.6B text encoder")

            # Anima uses base Qwen3 (not instruct) — tokenize directly, no chat template.
            # A safety cap is applied to prevent GPU OOM on extremely long prompts.
            text_inputs = tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=QWEN3_MAX_SEQ_LEN,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            if not isinstance(text_input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
                raise TypeError("Tokenizer returned unexpected types.")

            if text_input_ids.shape[-1] == QWEN3_MAX_SEQ_LEN:
                logger.warning(
                    f"Prompt was truncated to {QWEN3_MAX_SEQ_LEN} tokens. "
                    "Consider shortening the prompt for best results."
                )

            # Ensure at least 1 token (empty prompts produce 0 tokens with padding=False)
            if text_input_ids.shape[-1] == 0:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                text_input_ids = torch.tensor([[pad_id]])
                attention_mask = torch.tensor([[1]])

            # Get last hidden state from Qwen3 (final layer output)
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

            # Use last hidden state — only real tokens, no padding
            qwen3_embeds = outputs.hidden_states[-1][0]  # Shape: (seq_len, 1024)

        # --- Step 2: Tokenize with bundled T5-XXL tokenizer (IDs only, no model) ---
        context.util.signal_progress("Tokenizing with T5-XXL")
        t5_tokenizer = load_bundled_t5_tokenizer()
        t5_tokens = t5_tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=T5_MAX_SEQ_LEN,
            return_tensors="pt",
        )
        t5xxl_ids = t5_tokens.input_ids[0]  # Shape: (seq_len,)

        return qwen3_embeds, t5xxl_ids, None

    def _encode_qwen3_5(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the prompt with Qwen3.5 the way Anima-3.8B's reference encoder does.

        Returns:
            Tuple of (states, mask), on the CPU.
            - states: Shape (num_layers, seq_len, 2560), one row per layer in `QWEN35_LAYER_INDICES`.
            - mask: Shape (seq_len,), True for real tokens.
        """
        assert self.qwen3_5_encoder is not None
        context.util.signal_progress("Running Qwen3.5 text encoder")
        tokenizer_info = context.models.load(self.qwen3_5_encoder.tokenizer)
        with tokenizer_info.model_on_device() as (_, tokenizer):
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}.")
            # No template, no special tokens -- the prompt's tokens as they are.
            token_ids = tokenizer.encode(self.prompt, add_special_tokens=False)
            pad_token_id = tokenizer.pad_token_id
        if len(token_ids) > QWEN3_5_MAX_SEQ_LEN:
            logger.warning(f"Prompt was truncated to {QWEN3_5_MAX_SEQ_LEN} tokens for the Qwen3.5 encoder.")
            token_ids = token_ids[:QWEN3_5_MAX_SEQ_LEN]
        # An empty prompt becomes one padding token that nothing attends to, as in the reference, so the
        # connector adds no Qwen3.5 signal for it. (The reference also masks every token from the first
        # occurrence of id 151643 on -- the Qwen3 padding id, which in Qwen3.5's vocabulary is the
        # ordinary token " 내용". That is not reproduced.)
        is_empty = not token_ids
        if is_empty:
            if pad_token_id is None:
                raise ValueError("The Qwen3.5 tokenizer has no padding token to encode an empty prompt with.")
            token_ids = [pad_token_id]
        mask = torch.full((len(token_ids),), not is_empty, dtype=torch.bool)

        # Imported here: transformers' qwen3_5 modeling module costs ~0.25 s, which every app start would
        # otherwise pay for a node that only Anima-3.8B uses.
        from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder

        encoder_info = context.models.load(self.qwen3_5_encoder.text_encoder)
        with encoder_info.model_on_device() as (_, encoder):
            if not isinstance(encoder, Qwen35Encoder):
                raise TypeError(f"Expected a Qwen3.5 encoder, got {type(encoder).__name__}.")
            input_ids = torch.tensor([token_ids], device=encoder_info.compute_device)
            # The connector was trained on what Anima-3.8B's encoder checkpoint computes, and that file
            # ships its last tapped layer without the MLP.
            states = encoder(input_ids, QWEN35_LAYER_INDICES, last_layer_attention_only=True)
            stacked = torch.cat(states, dim=0).detach().to("cpu")
        return stacked, mask

    def _lora_iterator(self, context: InvocationContext) -> Iterator[PatchSpec]:
        """Iterate over LoRA models to apply to the Qwen3 text encoder."""
        for lora in self.qwen3_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight, lora_info.model_in_ram())
