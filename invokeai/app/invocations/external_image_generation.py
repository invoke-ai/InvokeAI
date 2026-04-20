from typing import TYPE_CHECKING, Any, ClassVar, Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation
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

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices


class BaseExternalImageGenerationInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generate images using an external provider."""

    provider_id: ClassVar[str | None] = None

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.main_model,
        ui_model_base=[BaseModelType.External],
        ui_model_type=[ModelType.ExternalImageGenerator],
        ui_model_format=[ModelFormat.ExternalApi],
    )
    mode: ExternalGenerationMode = InputField(
        default="txt2img",
        description="Generation mode. Not all modes are supported by every model; unsupported modes raise at runtime.",
    )
    prompt: str = InputField(description="Prompt")
    seed: int | None = InputField(default=None, description=FieldDescriptions.seed)
    num_images: int = InputField(default=1, gt=0, description="Number of images to generate")
    width: int = InputField(default=1024, gt=0, description=FieldDescriptions.width)
    height: int = InputField(default=1024, gt=0, description=FieldDescriptions.height)
    image_size: str | None = InputField(default=None, description="Image size preset (e.g. 1K, 2K, 4K)")
    init_image: ImageField | None = InputField(default=None, description="Init image for img2img/inpaint")
    mask_image: ImageField | None = InputField(default=None, description="Mask image for inpaint")
    reference_images: list[ImageField] = InputField(default=[], description="Reference images")

    def _build_provider_options(self) -> dict[str, Any] | None:
        """Override in provider-specific subclasses to pass extra options."""
        return None

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        model_config = context.models.get_config(self.model)
        if not isinstance(model_config, ExternalApiModelConfig):
            raise ValueError("Selected model is not an external API model")

        if self.provider_id is not None and model_config.provider_id != self.provider_id:
            raise ValueError(
                f"Selected model provider '{model_config.provider_id}' does not match node provider '{self.provider_id}'"
            )

        init_image = None
        if self.init_image is not None:
            init_image = context.images.get_pil(self.init_image.image_name, mode="RGB")

        mask_image = None
        if self.mask_image is not None:
            mask_image = context.images.get_pil(self.mask_image.image_name, mode="L")

        reference_images: list[ExternalReferenceImage] = []
        for image_field in self.reference_images:
            reference_image = context.images.get_pil(image_field.image_name, mode="RGB")
            reference_images.append(ExternalReferenceImage(image=reference_image))

        request = ExternalGenerationRequest(
            model=model_config,
            mode=self.mode,
            prompt=self.prompt,
            seed=self.seed,
            num_images=self.num_images,
            width=self.width,
            height=self.height,
            image_size=self.image_size,
            init_image=init_image,
            mask_image=mask_image,
            reference_images=reference_images,
            metadata=self._build_request_metadata(),
            provider_options=self._build_provider_options(),
        )

        result = context._services.external_generation.generate(request)

        outputs: list[ImageField] = []
        for generated in result.images:
            metadata = self._build_output_metadata(model_config, result, generated.seed)
            image_dto = context.images.save(image=generated.image, metadata=metadata)
            outputs.append(ImageField(image_name=image_dto.image_name))

        return ImageCollectionOutput(collection=outputs)

    def invoke_internal(self, context: InvocationContext, services: "InvocationServices") -> BaseInvocationOutput:
        """Override default cache behavior so cache hits produce new gallery entries.

        The standard invocation cache returns the cached output (with stale image_name
        references) without re-running invoke(), which means no new images are saved
        to the gallery on repeat invokes. For external API nodes — where the API call
        is the expensive part — we want cache hits to skip the API call but still
        produce fresh gallery entries by copying the cached images.
        """
        if services.configuration.node_cache_size == 0 or not self.use_cache:
            return super().invoke_internal(context, services)

        key = services.invocation_cache.create_key(self)
        cached_value = services.invocation_cache.get(key)
        if cached_value is None:
            services.logger.debug(f'Invocation cache miss for type "{self.get_type()}": {self.id}')
            output = self.invoke(context)
            services.invocation_cache.save(key, output)
            return output

        services.logger.debug(f'Invocation cache hit for type "{self.get_type()}": {self.id}, duplicating images')
        if not isinstance(cached_value, ImageCollectionOutput):
            return cached_value

        outputs: list[ImageField] = []
        for image_field in cached_value.collection:
            cached_image = context.images.get_pil(image_field.image_name, mode="RGB")
            image_dto = context.images.save(image=cached_image)
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

        if self.image_size is not None:
            metadata["image_size"] = self.image_size

        provider_request_id = getattr(result, "provider_request_id", None)
        if provider_request_id:
            metadata["external_request_id"] = provider_request_id

        provider_metadata = getattr(result, "provider_metadata", None)
        if provider_metadata:
            metadata["external_provider_metadata"] = provider_metadata

        if image_seed is not None:
            metadata["external_seed"] = image_seed

        metadata.update(self._build_output_provider_metadata())

        if not metadata:
            return None
        return MetadataField(root=metadata)

    def _build_output_provider_metadata(self) -> dict[str, Any]:
        """Override in provider-specific subclasses to add recall-relevant fields to the image metadata."""
        return {}


@invocation(
    "openai_image_generation",
    title="OpenAI Image Generation",
    tags=["external", "generation", "openai"],
    category="image",
    version="1.0.0",
)
class OpenAIImageGenerationInvocation(BaseExternalImageGenerationInvocation):
    """Generate images using an OpenAI-hosted external model."""

    provider_id = "openai"

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.main_model,
        ui_model_base=[BaseModelType.External],
        ui_model_type=[ModelType.ExternalImageGenerator],
        ui_model_format=[ModelFormat.ExternalApi],
        ui_model_provider_id=["openai"],
    )

    # OpenAI's API has no img2img/inpaint distinction — the edits endpoint is used
    # automatically when reference images are provided. Hide mode and init_image
    # (init_image is functionally identical to a reference image), and hide
    # mask_image since no OpenAI model supports inpainting.
    mode: ExternalGenerationMode = InputField(default="txt2img", description="Generation mode.", ui_hidden=True)
    init_image: ImageField | None = InputField(
        default=None, description="Init image (use reference_images instead)", ui_hidden=True
    )
    mask_image: ImageField | None = InputField(default=None, description="Mask image for inpaint", ui_hidden=True)

    quality: Literal["auto", "high", "medium", "low"] = InputField(default="auto", description="Output image quality")
    background: Literal["auto", "transparent", "opaque"] = InputField(
        default="auto", description="Background transparency handling"
    )
    input_fidelity: Literal["low", "high"] | None = InputField(
        default=None, description="Fidelity to source images (edits only)"
    )

    def _build_provider_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "quality": self.quality,
            "background": self.background,
        }
        if self.input_fidelity is not None:
            options["input_fidelity"] = self.input_fidelity
        return options

    def _build_output_provider_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "openai_quality": self.quality,
            "openai_background": self.background,
        }
        if self.input_fidelity is not None:
            metadata["openai_input_fidelity"] = self.input_fidelity
        return metadata


@invocation(
    "gemini_image_generation",
    title="Gemini Image Generation",
    tags=["external", "generation", "gemini"],
    category="image",
    version="1.0.0",
)
class GeminiImageGenerationInvocation(BaseExternalImageGenerationInvocation):
    """Generate images using a Gemini-hosted external model."""

    provider_id = "gemini"

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.main_model,
        ui_model_base=[BaseModelType.External],
        ui_model_type=[ModelType.ExternalImageGenerator],
        ui_model_format=[ModelFormat.ExternalApi],
        ui_model_provider_id=["gemini"],
    )

    # Gemini only supports txt2img — hide mode/init_image/mask_image fields
    # that are inherited from the base class but not usable with any Gemini model.
    mode: ExternalGenerationMode = InputField(default="txt2img", description="Generation mode.", ui_hidden=True)
    init_image: ImageField | None = InputField(
        default=None, description="Init image for img2img/inpaint", ui_hidden=True
    )
    mask_image: ImageField | None = InputField(default=None, description="Mask image for inpaint", ui_hidden=True)

    temperature: float | None = InputField(default=None, ge=0.0, le=2.0, description="Sampling temperature")

    def _build_provider_options(self) -> dict[str, Any] | None:
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        return options or None

    def _build_output_provider_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.temperature is not None:
            metadata["gemini_temperature"] = self.temperature
        return metadata
