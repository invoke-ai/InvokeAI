from contextlib import nullcontext

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3VLEncoderField
from invokeai.app.invocations.primitives import QwenImage21ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.qwen_image_2_1.text_encoding import encode_prompt, has_vision_tower
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
        description="Reference images to edit or draw from, in the order the denoise node receives their latents. "
        "Needs a Qwen-Image-2.1 Diffusers model's own encoder.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> QwenImage21ConditioningOutput:
        images = [to_reference(context.images.get_pil(field.image_name)) for field in self.reference_images]
        processor_id = self._processor_identifier(context) if images else None

        # The encoder first: loading it evicts unlocked small entries from the RAM cache, which would otherwise drop
        # the tokenizer and processor loaded before it.
        text_encoder_info = context.models.load(self.qwen3_vl_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_vl_encoder.tokenizer)
        processor_info = context.models.load(processor_id) if processor_id is not None else None
        # A GGUF or int8 encoder dequantizes one layer at a time; that transient is the working memory.
        dequant_bytes = peak_dequant_transient_bytes(
            text_encoder_info.model, TorchDevice.choose_bfloat16_safe_dtype(text_encoder_info.compute_device)
        )
        # Before the encoder moves to the GPU: a graph can pair this pipeline's tokenizer with another encoder.
        if images and not has_vision_tower(text_encoder_info.model):
            raise ValueError(_NO_VISION_TOWER)

        context.util.signal_progress("Running Qwen3-VL text encoder")
        with (
            tokenizer_info as tokenizer,
            processor_info.model_on_device() if processor_info is not None else nullcontext((None, None)) as (
                _,
                processor,
            ),
            text_encoder_info.model_on_device(working_mem_bytes=dequant_bytes) as (_, text_encoder),
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

    def _processor_identifier(self, context: InvocationContext) -> ModelIdentifierField:
        """The image processor of the model the tokenizer came from, which has to be a Qwen-Image-2.1 pipeline."""
        tokenizer = self.qwen3_vl_encoder.tokenizer
        config = context.models.get_config(tokenizer)
        if (
            config.base is not BaseModelType.QwenImage21
            or config.type is not ModelType.Main
            or config.format is not ModelFormat.Diffusers
        ):
            raise ValueError(_NO_VISION_TOWER)
        return tokenizer.model_copy(update={"submodel_type": SubModelType.Processor})
