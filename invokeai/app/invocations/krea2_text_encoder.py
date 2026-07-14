from contextlib import ExitStack
from typing import Iterator, Tuple

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3VLEncoderField
from invokeai.app.invocations.primitives import Krea2ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.sampling_utils import (
    KREA2_MAX_SEQ_LEN,
    KREA2_NUM_SUFFIX_TOKENS,
    KREA2_SELECT_LAYERS,
    KREA2_START_IDX,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_QWEN3VL_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

# Prompt template from diffusers Krea2Pipeline.get_text_hidden_states. The prefix (a system turn that
# instructs the model to describe the image) is the same "generate" template used by Qwen-Image, which
# is why the first KREA2_START_IDX (34) tokens are dropped from the encoder output.
_KREA2_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
_KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


@invocation(
    "krea2_text_encoder",
    title="Prompt - Krea-2",
    tags=["prompt", "conditioning", "krea2", "krea-2"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2TextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Krea-2 using the Qwen3-VL text encoder.

    The encoder taps 12 decoder hidden-state layers and stacks them per token, producing a 4D
    conditioning tensor (B, seq, 12, hidden) that the Krea-2 transformer's text-fusion stage consumes.
    """

    prompt: str = InputField(description="Text prompt describing the desired image.", ui_component=UIComponent.Textarea)
    qwen3_vl_encoder: Qwen3VLEncoderField = InputField(
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.qwen3_vl_encoder,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2ConditioningOutput:
        prompt_embeds, prompt_mask = self._encode(context)
        prompt_embeds = prompt_embeds.detach().to("cpu")
        prompt_mask = prompt_mask.detach().to("cpu") if prompt_mask is not None else None

        conditioning_data = ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return Krea2ConditioningOutput.build(conditioning_name)

    def _encode(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)

        # diffusers tokenizes (prefix + prompt) and the assistant-turn suffix separately, then
        # concatenates - so the suffix always survives truncation. Building one string and truncating it
        # (right-truncation) drops the suffix for long (>~500-token) prompts, corrupting the trained token
        # layout that the fixed prefix-drop (KREA2_START_IDX) and suffix accounting depend on.
        body_text = _KREA2_PREFIX + self.prompt
        # Reserve room for the suffix (diffusers: max_sequence_length + start_idx - num_suffix_tokens).
        body_max_length = KREA2_MAX_SEQ_LEN + KREA2_START_IDX - KREA2_NUM_SUFFIX_TOKENS

        context.util.signal_progress("Running Qwen3-VL text encoder")

        with ExitStack() as exit_stack:
            tokenizer = exit_stack.enter_context(tokenizer_info)
            (cached_weights, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            device = get_effective_device(text_encoder)

            # Apply any Qwen3-VL text-encoder LoRA patches (smart/sidecar patching, fp8-aware). Without
            # this, the encoder portion of a Krea-2 LoRA would be silently ignored.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=KREA2_LORA_QWEN3VL_PREFIX,
                    dtype=TorchDevice.choose_bfloat16_safe_dtype(device),
                    cached_weights=cached_weights,
                )
            )

            body_inputs = tokenizer(
                body_text,
                max_length=body_max_length,
                truncation=True,
                return_tensors="pt",
            )
            # Append the suffix AFTER truncation so it can never be cut. add_special_tokens=False keeps it
            # to exactly the assistant-turn tokens (no extra BOS), matching the reference token layout.
            suffix_inputs = tokenizer(_KREA2_SUFFIX, add_special_tokens=False, return_tensors="pt")
            input_ids = torch.cat([body_inputs.input_ids, suffix_inputs.input_ids], dim=1).to(device=device)
            attention_mask = torch.cat([body_inputs.attention_mask, suffix_inputs.attention_mask], dim=1).to(
                device=device
            )

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            # Some VL models nest the language-model output; fall back to that if needed.
            hidden_states_tuple = getattr(outputs, "hidden_states", None)
            if hidden_states_tuple is None:
                lm_output = getattr(outputs, "language_model_outputs", None)
                hidden_states_tuple = getattr(lm_output, "hidden_states", None)
            if hidden_states_tuple is None:
                raise RuntimeError("Qwen3-VL encoder did not return hidden_states; cannot build Krea-2 conditioning.")

            # Stack the selected layers along a new layer axis: (B, seq, 12, hidden).
            stacked = torch.stack([hidden_states_tuple[i] for i in KREA2_SELECT_LAYERS], dim=2)

            # Drop the system-prompt prefix tokens.
            prompt_embeds = stacked[:, KREA2_START_IDX:]
            prompt_mask = attention_mask[:, KREA2_START_IDX:].bool()

            # Match the device-safe compute dtype used by the denoise loop (falls back from bf16 to
            # fp16/fp32 on devices without bf16 support) rather than forcing bfloat16.
            prompt_embeds = prompt_embeds.to(dtype=TorchDevice.choose_bfloat16_safe_dtype(device))

        # If every token is valid (no padding), the mask is unnecessary.
        if prompt_mask is not None and bool(prompt_mask.all()):
            prompt_mask = None

        return prompt_embeds, prompt_mask

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over the LoRA models to apply to the Qwen3-VL text encoder."""
        for lora in self.qwen3_vl_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
