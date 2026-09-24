from contextlib import ExitStack
from typing import Iterator

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3VLEncoderField
from invokeai.app.invocations.primitives import Krea2ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.text_encoding import encode_krea2_prompt
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_QWEN3VL_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "krea2_text_encoder",
    title="Prompt - Krea-2",
    tags=["prompt", "conditioning", "krea2", "krea-2"],
    category="conditioning",
    version="1.1.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class Krea2TextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Krea-2 using the Qwen3-VL text encoder.

    The encoder taps 12 decoder hidden-state layers and stacks them per token, producing a 4D
    conditioning tensor (B, seq, 12, hidden) that the Krea-2 transformer's text-fusion stage consumes.
    """

    prompt: str = InputField(description="Text prompt describing the desired image.", ui_component=UIComponent.Textarea)
    mask: TensorField | None = InputField(
        default=None,
        description="A mask defining the image region that this conditioning prompt applies to.",
        input=Input.Connection,
    )
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
        return Krea2ConditioningOutput.build(conditioning_name, mask=self.mask)

    def _encode(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)

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

            prompt_embeds, prompt_mask, _ = encode_krea2_prompt(self.prompt, tokenizer, text_encoder)

        return prompt_embeds, prompt_mask

    def _lora_iterator(self, context: InvocationContext) -> Iterator[PatchSpec]:
        """Iterate over the LoRA models to apply to the Qwen3-VL text encoder."""
        for lora in self.qwen3_vl_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            yield (lora_info.model, lora.weight, lora_info.model_in_ram())
