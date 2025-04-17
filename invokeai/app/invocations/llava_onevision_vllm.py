from typing import Any

import torch
from PIL.Image import Image
from pydantic import field_validator
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, UIComponent, UIType
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.llava_onevision_pipeline import LlavaOnevisionPipeline
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "llava_onevision_vllm",
    title="LLaVA OneVision VLLM",
    tags=["vllm"],
    category="vllm",
    version="1.0.0",
    classification=Classification.Beta,
)
class LlavaOnevisionVllmInvocation(BaseInvocation):
    """Run a LLaVA OneVision VLLM model."""

    images: list[ImageField] | ImageField | None = InputField(default=None, max_length=3, description="Input image.")
    prompt: str = InputField(
        default="",
        description="Input text prompt.",
        ui_component=UIComponent.Textarea,
    )
    vllm_model: ModelIdentifierField = InputField(
        title="LLaVA Model Type",
        description=FieldDescriptions.vllm_model,
        ui_type=UIType.LlavaOnevisionModel,
    )

    @field_validator("images", mode="before")
    def listify_images(cls, v: Any) -> list:
        if v is None:
            return v
        if not isinstance(v, list):
            return [v]
        return v

    def _get_images(self, context: InvocationContext) -> list[Image]:
        if self.images is None:
            return []

        image_fields = self.images if isinstance(self.images, list) else [self.images]
        return [context.images.get_pil(image_field.image_name, "RGB") for image_field in image_fields]

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        images = self._get_images(context)
        model_config = context.models.get_config(self.vllm_model)

        with context.models.load(self.vllm_model).model_on_device() as (_, model):
            assert isinstance(model, LlavaOnevisionForConditionalGeneration)

            model_abs_path = context.models.get_absolute_path(model_config)
            processor = AutoProcessor.from_pretrained(model_abs_path, local_files_only=True)
            assert isinstance(processor, LlavaOnevisionProcessor)

            model = LlavaOnevisionPipeline(model, processor)
            output = model.run(
                prompt=self.prompt,
                images=images,
                device=TorchDevice.choose_torch_device(),
                dtype=TorchDevice.choose_torch_dtype(),
            )

        return StringOutput(value=output)
