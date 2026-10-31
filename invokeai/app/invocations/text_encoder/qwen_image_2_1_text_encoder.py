from contextlib import nullcontext

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3VLEncoderField
from invokeai.app.invocations.primitives import QwenImage21ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.architectures import FeaturesFacet, require
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.qwen_image_2_1.text_encoding import (
    attached_vision_tower,
    encode_prompt,
    sequence_length,
    vision_attention_bytes,
    working_memory_bytes,
)
from invokeai.backend.qwen_image_2_1.vae import to_reference
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImage21ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

_NO_VISION_TOWER = (
    "Reference images are read by the Qwen3-VL vision tower, which only the Qwen-Image-2.1 Diffusers model's own "
    "encoder brings. Clear the standalone encoder, or use a Qwen-Image-2.1 Diffusers model."
)


@invocation(
    "qwen_image_2_1_text_encoder",
    title="Prompt - Qwen-Image-2.1",
    tags=["prompt", "conditioning", "qwen_image_2_1", "qwen-image-2.1"],
    category="conditioning",
    version="1.1.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class QwenImage21TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for Qwen-Image-2.1 with its Qwen3-VL-8B encoder, with the reference images an edit reads.

    Each reference also goes to the denoise node, encoded by Image to Latents in its reference mode, in the same
    order.
    """

    prompt: str = InputField(description="Text prompt describing the desired image.", ui_component=UIComponent.Textarea)
    qwen3_vl_encoder: Qwen3VLEncoderField = InputField(
        title="Qwen3-VL Encoder", description=FieldDescriptions.qwen3_vl_encoder, input=Input.Connection
    )
    reference_images: list[ImageField] = InputField(
        default=[],
        description="Reference images to edit or draw from, in the order the denoise node receives their latents, "
        f"at most {require(BaseModelType.QwenImage21, FeaturesFacet).max_reference_images}. Needs a Qwen-Image-2.1 "
        "Diffusers model's own encoder.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> QwenImage21ConditioningOutput:
        max_references = require(BaseModelType.QwenImage21, FeaturesFacet).max_reference_images
        if len(self.reference_images) > max_references:
            raise ValueError(
                f"Qwen-Image-2.1 reads at most {max_references} reference images; this prompt has "
                f"{len(self.reference_images)}. Remove references until {max_references} remain."
            )
        images = [to_reference(context.images.get_pil(field.image_name)) for field in self.reference_images]
        tower_id, processor_id = self._edit_identifiers(context) if images else (None, None)

        # The encoder first: loading it evicts unlocked small entries from the RAM cache, which would otherwise drop
        # the tokenizer, tower and processor loaded after it.
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        tower_info = context.models.load(tower_id) if tower_id is not None else None
        processor_info = context.models.load(processor_id) if processor_id is not None else None
        if tower_info is not None and tower_info.compute_device != text_encoder_info.compute_device:
            raise RuntimeError(
                f"The vision tower runs on {tower_info.compute_device} and the encoder on "
                f"{text_encoder_info.compute_device}; an edit attaches one to the other, so they need one device."
            )
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(text_encoder_info.compute_device)
        # A GGUF or int8 encoder dequantizes one layer at a time; that transient is the working memory.
        dequant_bytes = peak_dequant_transient_bytes(text_encoder_info.model, compute_dtype)
        vision_bytes = (
            vision_attention_bytes(tower_info.model, images, tower_info.compute_device, compute_dtype)
            if tower_info is not None
            else 0
        )

        context.util.signal_progress("Running Qwen3-VL text encoder")
        with tokenizer_info as tokenizer:
            # Reference images make the sequence long, and its attention grows with the square of it: room the
            # model cache has to make, or the driver spills the encode to system memory and it crawls.
            num_tokens = sequence_length(tokenizer, self.prompt, images)
            working_memory = dequant_bytes + working_memory_bytes(num_tokens, compute_dtype.itemsize, vision_bytes)
            # The tower locks before the encoder, so the room the encoder reserves is still free once both are on
            # the device.
            with (
                processor_info.model_on_device() if processor_info is not None else nullcontext((None, None)) as (
                    _,
                    processor,
                ),
                tower_info.model_on_device() if tower_info is not None else nullcontext((None, None)) as (_, tower),
                text_encoder_info.model_on_device(working_mem_bytes=working_memory) as (_, text_encoder),
                attached_vision_tower(text_encoder, tower) if tower is not None else nullcontext(),
            ):
                encoding = encode_prompt(
                    text_encoder,
                    tokenizer,
                    self.prompt,
                    get_effective_device(text_encoder),
                    processor=processor,
                    images=images,
                )

        conditioning = QwenImage21ConditioningInfo(
            prompt_embeds=encoding.embeds.detach().cpu(),
            image_pad_mask=None if encoding.image_pad_mask is None else encoding.image_pad_mask.cpu(),
            reference_grids=encoding.reference_grids,
        )
        name = context.conditioning.save(ConditioningFieldData(conditionings=[conditioning]))
        return QwenImage21ConditioningOutput.build(name)

    def _edit_identifiers(self, context: InvocationContext) -> tuple[ModelIdentifierField, ModelIdentifierField]:
        """The vision tower and the image processor an edit adds, which only a Qwen-Image-2.1 pipeline brings.

        Checked on the configs, before anything loads: a graph can pair a standalone encoder with this pipeline's
        tokenizer, or the other way round.
        """
        for identifier in (self.qwen3_vl_encoder.tokenizer, self.qwen3_vl_encoder.text_encoder):
            config = context.models.get_config(identifier)
            if (
                config.base is not BaseModelType.QwenImage21
                or config.type is not ModelType.Main
                or config.format is not ModelFormat.Diffusers
            ):
                raise ValueError(_NO_VISION_TOWER)
        return (
            self.qwen3_vl_encoder.text_encoder.model_copy(update={"submodel_type": SubModelType.VisionEncoder}),
            self.qwen3_vl_encoder.tokenizer.model_copy(update={"submodel_type": SubModelType.Processor}),
        )
