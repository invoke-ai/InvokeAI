"""Cloud model configurations for cloud-based image generation services."""

from typing import Any, Dict, List, Literal

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    CloudProviderType,
    ModelFormat,
    ModelType,
)


class NotACloudModelError(Exception):
    """Raised when attempting to create a cloud model config from a non-cloud source."""


class CloudModelConfig(Config_Base):
    """Base configuration for cloud-based image generation models."""

    type: Literal[ModelType.Main] = ModelType.Main
    base: Literal[BaseModelType.CloudAPI] = BaseModelType.CloudAPI
    format: Literal[ModelFormat.CloudREST] = ModelFormat.CloudREST

    # Cloud-specific fields
    provider: CloudProviderType = Field(description="Cloud provider type")
    cloud_model_id: str = Field(description="Model ID on cloud service")

    # Provider-specific settings (flexible JSON blob for provider-specific configs)
    provider_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configuration parameters",
    )

    @classmethod
    def from_model_on_disk(
        cls,
        mod: ModelOnDisk,
        override_fields: dict[str, Any],
    ) -> "CloudModelConfig":
        """
        Cloud models are not discovered from disk. They must be registered manually.

        Raises NotACloudModelError to indicate this is not a cloud model source.
        """
        raise NotACloudModelError(
            "Cloud models cannot be auto-discovered from disk. Please register them manually via API or UI."
        )


class GeminiFlashImageConfig(CloudModelConfig):
    """Google Gemini 2.5 Flash Image configuration.

    Official API Documentation:
    https://ai.google.dev/gemini-api/docs/image-generation

    Pricing: $0.039 per image (1290 output tokens)
    """

    provider: Literal[CloudProviderType.GoogleGemini] = CloudProviderType.GoogleGemini
    cloud_model_id: Literal["gemini-2.5-flash-image"] = "gemini-2.5-flash-image"

    # Supported aspect ratios per official spec
    supported_aspect_ratios: List[str] = Field(
        default=[
            "1:1",    # 1024x1024
            "3:2",    # 1536x1024
            "2:3",    # 1024x1536
            "3:4",    # 1152x1536
            "4:3",    # 1536x1152
            "4:5",    # 1024x1280
            "5:4",    # 1280x1024
            "9:16",   # 576x1024
            "16:9",   # 1024x576
            "21:9",   # 1344x576
        ],
        description="Supported aspect ratios for Gemini 2.5 Flash Image",
    )

    # API endpoint
    api_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Google Gemini API base URL",
    )

    # Model capabilities
    supports_seed: bool = Field(default=True, description="Model supports deterministic generation with seed")
    supports_negative_prompt: bool = Field(
        default=False, description="Model does not support negative prompts"
    )
    max_prompt_length: int = Field(default=8192, description="Maximum prompt length in characters")


class ImagenUltraConfig(CloudModelConfig):
    """Google Imagen 4 Ultra configuration.

    Official API Documentation:
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api

    Pricing: $0.06 per image
    """

    provider: Literal[CloudProviderType.GoogleImagen] = CloudProviderType.GoogleImagen
    cloud_model_id: Literal["imagen-4.0-ultra-generate-001"] = "imagen-4.0-ultra-generate-001"

    # Supported aspect ratios (Imagen-specific)
    supported_aspect_ratios: List[str] = Field(
        default=[
            "1:1",    # 1024x1024
            "3:4",    # 768x1024
            "4:3",    # 1024x768
            "9:16",   # 576x1024
            "16:9",   # 1024x576
        ],
        description="Supported aspect ratios for Imagen 4 Ultra",
    )

    # Max images per request
    max_batch_size: int = Field(default=4, description="Maximum number of images per generation request")

    # Safety setting options
    safety_levels: List[str] = Field(
        default=[
            "block_low_and_above",
            "block_medium_and_above",
            "block_only_high",
        ],
        description="Available safety filter levels",
    )

    # Model capabilities
    supports_seed: bool = Field(default=True, description="Model supports deterministic generation with seed")
    supports_synthid_watermark: bool = Field(
        default=True, description="Model includes SynthID watermark in generated images"
    )
    supports_prompt_enhancement: bool = Field(
        default=True, description="Model supports LLM-based prompt enhancement"
    )
    max_prompt_length: int = Field(default=480, description="Maximum prompt length in tokens")
    max_resolution: int = Field(default=2048, description="Maximum image resolution (2K)")


class OpenAIImageConfig(CloudModelConfig):
    """OpenAI DALL-E 3 / GPT Image 1 configuration.

    Official API Documentation:
    https://platform.openai.com/docs/guides/image-generation

    Pricing: DALL-E 3 - $0.04-$0.12 per image depending on quality and size
    """

    provider: Literal[CloudProviderType.OpenAI] = CloudProviderType.OpenAI

    # Model ID can be dall-e-3, dall-e-2, or gpt-image-1 (when available)
    cloud_model_id: str = Field(
        default="dall-e-3",
        description="OpenAI model ID (dall-e-3, dall-e-2, or gpt-image-1)",
    )

    # Supported sizes (DALL-E 3)
    supported_sizes: List[str] = Field(
        default=[
            "1024x1024",
            "1792x1024",
            "1024x1792",
        ],
        description="Supported image sizes",
    )

    # Quality options (DALL-E 3 only)
    quality_options: List[str] = Field(
        default=["standard", "hd"],
        description="Image quality options",
    )

    # Style options (DALL-E 3 only)
    style_options: List[str] = Field(
        default=["vivid", "natural"],
        description="Image style options",
    )

    # Model capabilities
    supports_seed: bool = Field(default=False, description="DALL-E 3 does not support seed parameter")
    supports_revised_prompt: bool = Field(
        default=True, description="Model returns revised/enhanced prompt used for generation"
    )
    max_batch_size: int = Field(default=1, description="DALL-E 3 generates one image at a time")
    max_prompt_length: int = Field(default=4000, description="Maximum prompt length in characters")
