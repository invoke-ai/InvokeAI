import re
from contextlib import ExitStack
from typing import Iterator

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    Krea2ConditioningField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.metadata import extract_encoder_model, extract_loras_metadata
from invokeai.app.invocations.model import Qwen3VLEncoderField
from invokeai.app.invocations.primitives import Krea2ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.text_encoding import (
    KREA2_MAX_REFERENCE_IMAGES,
    encode_krea2_prompt,
    encode_krea2_prompt_with_images,
    prepare_vlm_reference,
    tokenize_krea2_prompt_with_images,
    working_memory_bytes,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.model_manager.taxonomy import ModelFormat, SubModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_QWEN3VL_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Krea2ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

# The loader drops the Qwen3-VL visual tower from the text encoder: text-only encoding never runs it, and
# reference images run a separately loaded tower, so in the encoder its weights would be dead resident
# bytes. An adapter can still carry layers for it -- a Krea-2 LoRA's encoder keys are
# whatever followed `text_encoder.`, and the converter does emit `lora_qwen3vl-visual.*` -- and those
# layers now resolve to no module. Left to the patcher that is one "Failed to find module for LoRA
# layer key" per layer per generation (~108 lines on the 4B, whose vision tower has 27 blocks),
# reading as though the adapter were broken. It is not, and it never did anything: patching a module
# that `forward` does not reach cannot change an image. With reference images the tower is a separate cache
# entry lent to the encoder only after the patches are in place, so its layers stay unpatched there too:
# ai-toolkit trains edit LoRAs with the encoder frozen.
_VISUAL_TOWER_LORA_LAYERS = re.compile(rf"^{re.escape(KREA2_LORA_QWEN3VL_PREFIX)}visual\.")

_NO_VISION_TOWER = (
    "Reference images are read by the Qwen3-VL vision tower, and the GGUF encoder '{name}' has none. Select the "
    "Vision Tower model (the 4B llama.cpp mmproj file) beside it, or use a Diffusers Krea-2 model's own encoder or "
    "a standalone non-GGUF Qwen3-VL 4B encoder."
)


@invocation(
    "krea2_text_encoder",
    title="Prompt - Krea-2",
    tags=["prompt", "conditioning", "krea2", "krea-2"],
    category="conditioning",
    version="1.3.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class Krea2TextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Krea-2 using the Qwen3-VL text encoder.

    The encoder taps 12 decoder hidden-state layers and stacks them per token, producing a 4D
    conditioning tensor (B, seq, 12, hidden) that the Krea-2 transformer's text-fusion stage consumes.

    With reference images (edit mode, for edit LoRAs) the Qwen3-VL vision tower reads them into the prompt,
    ahead of the text, and the conditioning has the prompt's natural length rather than 512 tokens.
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
    reference_images: list[ImageField] = InputField(
        default=[],
        max_length=KREA2_MAX_REFERENCE_IMAGES,
        description="Up to three reference images for an edit LoRA, read by the Qwen3-VL vision tower in this order "
        "(Picture 1, 2, 3). A GGUF encoder reads them through the selected Vision Tower model (the mmproj file).",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2ConditioningOutput:
        prompt_embeds, prompt_mask = self._encode(context)
        prompt_embeds = prompt_embeds.detach().to("cpu")
        prompt_mask = prompt_mask.detach().to("cpu") if prompt_mask is not None else None

        conditioning_data = ConditioningFieldData(
            conditionings=[
                Krea2ConditioningInfo(
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_mask,
                    num_reference_images=len(self.reference_images),
                )
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        loras = extract_loras_metadata(getattr(self.qwen3_vl_encoder, "loras", None))
        return Krea2ConditioningOutput(
            conditioning=Krea2ConditioningField(
                conditioning_name=conditioning_name,
                mask=self.mask,
                prompt=self.prompt,
                qwen3_vl_encoder=extract_encoder_model(self.qwen3_vl_encoder),
                loras=loras,
            )
        )

    def _encode(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor | None]:
        images = [prepare_vlm_reference(context.images.get_pil(field.image_name)) for field in self.reference_images]
        encoder_config = context.models.get_config(self.qwen3_vl_encoder.text_encoder)
        text_encoder_format = encoder_config.format
        # Chosen on the config, before anything heavy loads. A GGUF encoder has no tower of its own: it reads the
        # model the field names, the mmproj file (not a submodel of anything). Any other encoder carries its own.
        vision_identifier = None
        if images:
            field = self.qwen3_vl_encoder
            if text_encoder_format is not ModelFormat.GGUFQuantized:
                vision_identifier = field.text_encoder.model_copy(update={"submodel_type": SubModelType.VisionEncoder})
            elif field.vision_tower is not None:
                vision_identifier = field.vision_tower.model_copy(update={"submodel_type": None})
            else:
                raise ValueError(_NO_VISION_TOWER.format(name=encoder_config.name))
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        image_inputs = vision_info = None
        if vision_identifier is not None:
            # The processor and tokenizer are small and run on the CPU, so a reference the processor refuses (an
            # extreme aspect ratio) fails here, before either tower is read.
            processor_info = context.models.load(
                self.qwen3_vl_encoder.tokenizer.model_copy(update={"submodel_type": SubModelType.Processor})
            )
            with processor_info as processor, tokenizer_info as tokenizer:
                image_inputs = tokenize_krea2_prompt_with_images(self.prompt, images, processor, tokenizer)
            # Before the encoder: the loader refuses an encoder without a tower here, ahead of reading
            # the far larger language tower for nothing. The tower is its own cache entry, so text-only prompts
            # never pay for it.
            vision_info = context.models.load(vision_identifier)
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)
        if vision_info is not None and vision_info.compute_device != text_encoder_info.compute_device:
            raise RuntimeError(
                f"The vision tower runs on {vision_info.compute_device} and the encoder on "
                f"{text_encoder_info.compute_device}; an edit attaches one to the other, so they need one device. "
                "Give both models the same CPU Only setting in the Model Manager."
            )
        # An nvfp4 or GGUF build dequantizes each packed Linear per forward, a transient its resident size does
        # not cover. Read from the unlocked model, before the lock the reservation applies to; zero for
        # other builds.
        working_mem_bytes = peak_dequant_transient_bytes(
            text_encoder_info.model, TorchDevice.choose_bfloat16_safe_dtype(text_encoder_info.compute_device)
        )
        if image_inputs is not None:
            # The references lengthen the sequence, and attention grows with its square. The image path does not
            # truncate, so this is the prompt's natural length.
            working_mem_bytes += working_memory_bytes(image_inputs["input_ids"].shape[1])

        context.util.signal_progress("Running Qwen3-VL text encoder")

        with ExitStack() as exit_stack:
            # The image path has tokenized already.
            text_tokenizer = exit_stack.enter_context(tokenizer_info) if image_inputs is None else None
            (cached_weights, text_encoder) = exit_stack.enter_context(
                text_encoder_info.model_on_device(working_mem_bytes=working_mem_bytes)
            )
            device = get_effective_device(text_encoder)

            # Apply any Qwen3-VL text-encoder LoRA patches (smart/sidecar patching, fp8-aware). Without
            # this, the encoder portion of a Krea-2 LoRA would be silently ignored.
            #
            # `force_sidecar_patching` is not optional for a quantized encoder: the patcher does not
            # detect quantization itself and expects the caller to say so. A GGML-quantized Linear
            # reports `dtype=uint8` (so the fp8 check misses it) and a packed `nelement()`, so direct
            # patching compares the LoRA against the packed byte count, drops every layer, and blames
            # the LoRA in the warning. It is also VRAM-dependent: with the layers on CPU the patcher
            # picks the sidecar anyway, so the same graph would silently produce different images
            # under memory pressure.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=KREA2_LORA_QWEN3VL_PREFIX,
                    dtype=TorchDevice.choose_bfloat16_safe_dtype(device),
                    cached_weights=cached_weights,
                    force_sidecar_patching=requires_sidecar_patching(text_encoder, text_encoder_format),
                    suppress_warning_layers=_VISUAL_TOWER_LORA_LAYERS,
                )
            )

            if text_tokenizer is not None:
                prompt_embeds, prompt_mask, _ = encode_krea2_prompt(self.prompt, text_tokenizer, text_encoder)
            else:
                assert image_inputs is not None and vision_info is not None
                # Locked after the encoder and released before it (the stack unwinds in reverse), and the
                # encode undoes the swap before returning: neither cache entry may be moved while the tower is
                # part of the encoder's module tree. With the encoder's reservation: the cache counts the locked
                # encoder as in use, so the tower is placed only in VRAM beyond the room the encoder's forward
                # needs (the default reservation would let it take that room). Adding the tower's size to the
                # encoder's reservation instead would count a tower that is already resident twice.
                (_, vision_tower) = exit_stack.enter_context(
                    vision_info.model_on_device(working_mem_bytes=working_mem_bytes)
                )
                prompt_embeds, prompt_mask = encode_krea2_prompt_with_images(image_inputs, text_encoder, vision_tower)

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
