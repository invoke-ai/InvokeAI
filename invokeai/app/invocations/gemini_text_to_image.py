"""Google Gemini 2.5 Flash Image text-to-image invocation."""

import asyncio
from io import BytesIO
from typing import Literal, Optional

from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext
from invokeai.app.invocations.fields import ImageField, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.cloud_providers.provider_base import CloudGenerationRequest
from invokeai.app.services.images.images_common import ImageCategory


@invocation(
    "gemini_text_to_image",
    title="Gemini 2.5 Flash - Text to Image",
    tags=["image", "cloud", "gemini", "google", "text2img"],
    category="image",
    version="1.0.0",
)
class GeminiTextToImageInvocation(BaseInvocation):
    """Generate images using Google Gemini 2.5 Flash Image.

    Google Gemini 2.5 Flash Image is a state-of-the-art cloud-based image generation model.
    It supports 10 different aspect ratios and deterministic generation with seeds.

    Official Documentation: https://ai.google.dev/gemini-api/docs/image-generation
    Pricing: $0.039 per image
    """

    # Model selection
    model: ModelIdentifierField = InputField(
        description="Gemini 2.5 Flash Image model to use",
    )

    # Core parameters (exact API spec)
    prompt: str = InputField(
        description="Text description of the image to generate",
    )

    width: int = InputField(
        default=1024,
        ge=256,
        le=2048,
        multiple_of=64,
        description="Image width in pixels (will be adjusted to nearest supported aspect ratio)",
    )

    height: int = InputField(
        default=1024,
        ge=256,
        le=2048,
        multiple_of=64,
        description="Image height in pixels (will be adjusted to nearest supported aspect ratio)",
    )

    seed: Optional[int] = InputField(
        default=None,
        description="Random seed for deterministic generation (optional)",
    )

    # Aspect ratio helper (optional, for UI convenience)
    aspect_ratio: Optional[
        Literal["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    ] = InputField(
        default=None,
        description="Preset aspect ratio (overrides width/height if set)",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        """Execute Gemini image generation.

        Args:
            context: Invocation context with access to services

        Returns:
            ImageOutput with the generated image
        """
        # Apply aspect ratio preset if specified
        if self.aspect_ratio:
            # Official aspect ratio dimensions from Gemini API spec
            dimensions = {
                "1:1": (1024, 1024),
                "3:2": (1536, 1024),
                "2:3": (1024, 1536),
                "3:4": (1152, 1536),
                "4:3": (1536, 1152),
                "4:5": (1024, 1280),
                "5:4": (1280, 1024),
                "9:16": (576, 1024),
                "16:9": (1024, 576),
                "21:9": (1344, 576),
            }
            width, height = dimensions[self.aspect_ratio]
        else:
            width, height = self.width, self.height

        # Load cloud model (gets CloudModelWrapper with provider)
        loaded_model = context.models.load(self.model)

        with loaded_model as cloud_model:
            # Build request
            request = CloudGenerationRequest(
                prompt=self.prompt, width=width, height=height, seed=self.seed, num_images=1
            )

            # Emit progress event (cloud generation is single-step)
            context.util.events.emit_generator_progress(
                graph_execution_state_id=context.graph_execution_state_id,
                node_id=self.id,
                source_node_id=self.id,
                step=0,
                total_steps=1,
                order=0,
                progress_image=None,
            )

            # Call cloud API (async to sync conversion)
            # Gemini provider is async, but invocations are sync
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(cloud_model.generate(request))
            finally:
                loop.close()

            # Convert bytes to PIL Image
            image_bytes = response.images[0]
            pil_image = Image.open(BytesIO(image_bytes))

            # Save to InvokeAI image storage
            image_dto = context.images.create(
                image=pil_image,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
                metadata={
                    "prompt": self.prompt,
                    "model": "gemini-2.5-flash-image",
                    "provider": "google-gemini",
                    "width": width,
                    "height": height,
                    "seed": self.seed,
                    "aspect_ratio": self.aspect_ratio,
                    **response.metadata,
                },
            )

            # Return image output
            return ImageOutput.build(image_dto=image_dto)
