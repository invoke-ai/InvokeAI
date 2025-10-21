"""Abstract base classes for cloud model providers."""

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field


class CloudGenerationRequest(BaseModel):
    """Standardized request for cloud image generation across all providers.

    This abstraction allows different cloud providers to be used interchangeably
    through a common interface.
    """

    prompt: str = Field(description="Text description of the image to generate")
    width: int = Field(ge=256, le=4096, description="Image width in pixels")
    height: int = Field(ge=256, le=4096, description="Image height in pixels")
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic generation")
    num_images: int = Field(default=1, ge=1, le=10, description="Number of images to generate")

    # Optional parameters (provider-specific)
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt (if supported by provider)")
    guidance_scale: Optional[float] = Field(default=None, description="Guidance scale (if supported)")


class CloudGenerationResponse(BaseModel):
    """Standardized response from cloud image generation.

    Contains the generated images as raw bytes, along with metadata about
    the generation process.
    """

    images: List[bytes] = Field(description="Generated images as raw bytes (PNG/JPEG)")
    metadata: dict = Field(description="Generation metadata (model, parameters, etc.)")
    provider_response: dict = Field(description="Raw API response for debugging/logging")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class CloudModelProviderBase(ABC):
    """Abstract base class for cloud model providers.

    Each cloud provider (Google, OpenAI, etc.) should implement this interface.
    This ensures consistency across different cloud APIs and makes it easy to
    add new providers in the future.
    """

    def __init__(self, api_key: str, config: dict):
        """Initialize the cloud provider.

        Args:
            api_key: API key for authentication
            config: Provider-specific configuration settings
        """
        self.api_key = api_key
        self.config = config

    @abstractmethod
    async def generate_image(self, request: CloudGenerationRequest) -> CloudGenerationResponse:
        """Generate image(s) from the cloud API.

        Args:
            request: Standardized generation request

        Returns:
            CloudGenerationResponse with generated images and metadata

        Raises:
            Exception: If API call fails or returns an error
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that the API credentials are working.

        Returns:
            True if credentials are valid, False otherwise
        """
        pass
