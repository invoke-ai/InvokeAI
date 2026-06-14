"""FLUX.2 [dev] text encoder invocation.

FLUX.2 [dev] uses the BFL "cow-mistral3-small" 30-layer Mistral distillation as
its sole text encoder (sometimes referred to as "Mistral Small 3" in BFL's
documentation, but the shipped weights are the 30-layer cow variant — upstream
40-layer Mistral Small 3.1 / 3.2 does not work):

- A fixed system message biases the model toward structured image descriptions.
- The user prompt is wrapped in Mistral's chat template via the multimodal
  AutoProcessor.
- Three intermediate hidden states (layers 10, 20, 30) are stacked and flattened
  to produce a (B, seq, 3 * hidden_size) = (B, seq, 15360) tensor matching the
  FLUX.2 transformer's joint_attention_dim. For the 30-layer cow model those
  indices map to (1/3, 2/3, last) — exactly what BFL's joint attention was
  trained to consume.
"""

from contextlib import ExitStack
from typing import Any, Iterator, Literal, Optional, Tuple, cast

import torch
from transformers import PreTrainedModel

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import MistralEncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# System prompt used by the FLUX.2 [dev] reference pipeline. Byte-for-byte
# identical to ComfyUI's ``Flux2Tokenizer.llama_template`` — note the literal
# ``\n`` between "object" and "attribution"; that's part of the trained-against
# token sequence, not a formatting artifact.
FLUX2_DEV_SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object\nattribution and actions "
    "without speculation."
)

# Raw chat template fed straight to the tokenizer — matches Comfy's approach
# (no ``apply_chat_template`` indirection). ``[SYSTEM_PROMPT]`` / ``[INST]`` are
# special tokens in Mistral Small 3's Tekken vocab, so the encoder produces the
# exact token sequence BFL trained the joint attention against.
FLUX2_DEV_PROMPT_TEMPLATE = "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{prompt}[/INST]"

# Indices into hidden_states[] (hidden_states[0] is the embedding output) that
# FLUX.2 [dev]'s joint attention was trained to consume. ComfyUI uses these
# same indices for both the 30-layer cow distillation and the 40-layer Mistral
# Small 3; for cow they hit (1/3, 2/3, last), and the loader strips the final
# RMSNorm so the layer-30 readout is the raw post-layer-29 state.
DEV_EXTRACTION_LAYERS = (10, 20, 30)

# Default max sequence length for FLUX.2 [dev]. The reference pipeline caps at 512.
DEV_MAX_SEQ_LEN = 512


@invocation(
    "flux2_dev_text_encoder",
    title="Prompt - FLUX.2 [dev]",
    tags=["prompt", "conditioning", "flux", "flux2", "dev", "mistral"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DevTextEncoderInvocation(BaseInvocation):
    """Encode a prompt for FLUX.2 [dev] using its Mistral Small 3.1 text encoder."""

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    mistral_encoder: MistralEncoderField = InputField(
        title="Mistral Encoder",
        description=FieldDescriptions.mistral_encoder,
        input=Input.Connection,
    )
    max_seq_len: Literal[256, 512] = InputField(
        default=DEV_MAX_SEQ_LEN,
        description="Max sequence length for the Mistral encoder.",
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        with ExitStack() as exit_stack:
            mistral_embeds = self._encode_prompt(context, exit_stack)

            # FLUX.2 [dev] does not consume a pooled / CLIP-style embedding; we
            # reuse the FLUX conditioning structure (Klein does the same) and put
            # the Mistral hidden states in the `t5_embeds` slot, which the
            # FLUX.2 denoise loop already wires into `encoder_hidden_states`.
            conditioning_data = ConditioningFieldData(
                conditionings=[
                    FLUXConditioningInfo(
                        clip_embeds=torch.zeros(1, device=mistral_embeds.device, dtype=mistral_embeds.dtype),
                        t5_embeds=mistral_embeds,
                    )
                ]
            )
            conditioning_name = context.conditioning.save(conditioning_data)
            return FluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
            )

    def _encode_prompt(self, context: InvocationContext, exit_stack: ExitStack) -> torch.Tensor:
        text_encoder_info = context.models.load(self.mistral_encoder.text_encoder)
        (cached_weights, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())

        processor_info = context.models.load(self.mistral_encoder.tokenizer)
        (_, processor) = exit_stack.enter_context(processor_info.model_on_device())

        repaired_tensors = text_encoder_info.repair_required_tensors_on_device()
        device = get_effective_device(text_encoder)
        if repaired_tensors > 0:
            context.logger.warning(
                f"Recovered {repaired_tensors} required Mistral tensor(s) on {device} after a partial device mismatch."
            )

        # Apply any LoRAs attached to the text encoder.
        lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
        exit_stack.enter_context(
            LayerPatcher.apply_smart_model_patches(
                model=text_encoder,
                patches=self._lora_iterator(context),
                prefix=FLUX_LORA_T5_PREFIX,
                dtype=lora_dtype,
                cached_weights=cached_weights,
            )
        )

        context.util.signal_progress("Running Mistral text encoder (FLUX.2 [dev])")

        if not isinstance(text_encoder, PreTrainedModel):
            raise TypeError(
                f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}. "
                "The Mistral encoder model may be corrupted or incompatible."
            )

        # Build the raw FLUX.2 [dev] prompt template — matches ComfyUI's
        # `Flux2Tokenizer.llama_template.format(text)` byte-for-byte. `[SYSTEM_PROMPT]`,
        # `[/SYSTEM_PROMPT]`, `[INST]`, `[/INST]` are Tekken special tokens, so any of
        # the three processors we can land on (Pixtral/Mistral3 processor, plain HF
        # LlamaTokenizerFast, our embedded-Tekken adapter) emit the same sequence.
        text = FLUX2_DEV_PROMPT_TEMPLATE.format(system=FLUX2_DEV_SYSTEM_MESSAGE, prompt=self.prompt)

        # Comfy pads on the LEFT (`pad_left=True`), keeping the meaningful tokens
        # at the right edge of the sequence. HF processors expose this via the
        # `padding_side` attribute on their underlying tokenizer; we set it
        # explicitly so the call matches Comfy's behavior regardless of the
        # tokenizer's default. `processor` is typed as the `AnyModel` union;
        # narrow to `Any` for the duration of the tokenizer call.
        proc = cast(Any, processor)
        tokenizer = getattr(proc, "tokenizer", proc)
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        inputs = proc(
            text,
            return_tensors="pt",
            padding="max_length",
            padding_side="left",
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Mistral3ForConditionalGeneration wraps the LM under `.language_model`.
        # For pure text encoding, run that sub-module to skip the (unused) vision
        # tower and to avoid emitting a generation; for plain MistralModel /
        # MistralForCausalLM, run the model directly.
        forward_target = getattr(text_encoder, "language_model", None) or text_encoder

        outputs = forward_target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(
                "Mistral encoder did not return hidden_states. "
                "Ensure output_hidden_states=True is supported by this model."
            )
        num_hidden_states = len(outputs.hidden_states)  # = num_hidden_layers + 1 (embedding output)

        # Safety check: the model loaders only accept 30-layer cow weights, so
        # hidden_states[] should have ≥ 31 entries (embedding output + 30 layers).
        # Fall back to a scaled tuple only if a non-cow encoder somehow slipped
        # past the loaders, so we don't crash with an IndexError.
        if num_hidden_states - 1 < max(DEV_EXTRACTION_LAYERS):
            n = num_hidden_states - 1
            extraction_layers = (max(1, n // 3), max(1, (2 * n) // 3), n)
        else:
            extraction_layers = DEV_EXTRACTION_LAYERS

        stacked = torch.stack([outputs.hidden_states[i] for i in extraction_layers], dim=1)
        # stacked: (B, 3, seq, hidden_size) -> (B, seq, 3 * hidden_size)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        return prompt_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRAs to apply to the Mistral encoder."""
        for lora in self.mistral_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
