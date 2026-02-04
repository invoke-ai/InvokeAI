from typing import Any

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    MetadataField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageCollectionOutput
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalGenerationResult,
    ExternalReferenceImage,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalGenerationMode
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


@invocation(
    "external_image_generation",
    title="External Image Generation",
    tags=["external", "generation"],
    category="image",
    version="1.0.0",
)
class ExternalImageGenerationInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generate images using an external provider."""

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.main_model,
        ui_model_base=[BaseModelType.External],
        ui_model_type=[ModelType.ExternalImageGenerator],
        ui_model_format=[ModelFormat.ExternalApi],
    )
    mode: ExternalGenerationMode = InputField(default="txt2img", description="Generation mode")
    prompt: str = InputField(description="Prompt")
    negative_prompt: str | None = InputField(default=None, description="Negative prompt")
    seed: int | None = InputField(default=None, description=FieldDescriptions.seed)
    num_images: int = InputField(default=1, gt=0, description="Number of images to generate")
    width: int = InputField(default=1024, gt=0, description=FieldDescriptions.width)
    height: int = InputField(default=1024, gt=0, description=FieldDescriptions.height)
    steps: int | None = InputField(default=None, gt=0, description=FieldDescriptions.steps)
    guidance: float | None = InputField(default=None, ge=0, description="Guidance strength")
    init_image: ImageField | None = InputField(default=None, description="Init image for img2img/inpaint")
    mask_image: ImageField | None = InputField(default=None, description="Mask image for inpaint")
    reference_images: list[ImageField] = InputField(default=[], description="Reference images")
    reference_image_weights: list[float] | None = InputField(default=None, description="Reference image weights")
    reference_image_modes: list[str] | None = InputField(default=None, description="Reference image modes")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        model_config = context.models.get_config(self.model)
        if not isinstance(model_config, ExternalApiModelConfig):
            raise ValueError("Selected model is not an external API model")

        init_image = None
        if self.init_image is not None:
            init_image = context.images.get_pil(self.init_image.image_name, mode="RGB")

        mask_image = None
        if self.mask_image is not None:
            mask_image = context.images.get_pil(self.mask_image.image_name, mode="L")

        if self.reference_image_weights is not None and len(self.reference_image_weights) != len(self.reference_images):
            raise ValueError("reference_image_weights must match reference_images length")

        if self.reference_image_modes is not None and len(self.reference_image_modes) != len(self.reference_images):
            raise ValueError("reference_image_modes must match reference_images length")

        reference_images: list[ExternalReferenceImage] = []
        for index, image_field in enumerate(self.reference_images):
            reference_image = context.images.get_pil(image_field.image_name, mode="RGB")
            weight = None
            mode = None
            if self.reference_image_weights is not None:
                weight = self.reference_image_weights[index]
            if self.reference_image_modes is not None:
                mode = self.reference_image_modes[index]
            reference_images.append(ExternalReferenceImage(image=reference_image, weight=weight, mode=mode))

        request = ExternalGenerationRequest(
            model=model_config,
            mode=self.mode,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            num_images=self.num_images,
            width=self.width,
            height=self.height,
            steps=self.steps,
            guidance=self.guidance,
            init_image=init_image,
            mask_image=mask_image,
            reference_images=reference_images,
            metadata=self._build_request_metadata(),
        )

        result = context._services.external_generation.generate(request)

        outputs: list[ImageField] = []
        for generated in result.images:
            metadata = self._build_output_metadata(model_config, result, generated.seed)
            image_dto = context.images.save(image=generated.image, metadata=metadata)
            outputs.append(ImageField(image_name=image_dto.image_name))

        return ImageCollectionOutput(collection=outputs)

    def _build_request_metadata(self) -> dict[str, Any] | None:
        if self.metadata is None:
            return None
        return self.metadata.root

    def _build_output_metadata(
        self,
        model_config: ExternalApiModelConfig,
        result: ExternalGenerationResult,
        image_seed: int | None,
    ) -> MetadataField | None:
        metadata: dict[str, Any] = {}

        if self.metadata is not None:
            metadata.update(self.metadata.root)

        metadata.update(
            {
                "external_provider": model_config.provider_id,
                "external_model_id": model_config.provider_model_id,
            }
        )

        provider_request_id = getattr(result, "provider_request_id", None)
        if provider_request_id:
            metadata["external_request_id"] = provider_request_id

        provider_metadata = getattr(result, "provider_metadata", None)
        if provider_metadata:
            metadata["external_provider_metadata"] = provider_metadata

        if image_seed is not None:
            metadata["external_seed"] = image_seed

        if not metadata:
            return None
        return MetadataField(root=metadata)
