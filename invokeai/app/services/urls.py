from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class ImageUrls(BaseModel):
    """Contains URLs for an image."""

    image_url: str = Field(description="The URL of the image.")
    thumbnail_url: str = Field(description="The thumbnail URL of the image.")


class TensorUrl(BaseModel):
    """Contains URL for a tensor."""

    tensor_url: str = Field(description="The URL of the tensor.")


class URLServiceBase(ABC):
    """Responsible for building URLs for resources (eg images or tensors)."""

    @abstractmethod
    def get_image_urls(self, image_id: str) -> ImageUrls:
        """Gets the URL(s) for a resource."""
        pass


class LocalURLService(URLServiceBase):
    def __init__(self, base_url: str = "api/v1"):
        self._base_url = base_url

    def get_image_urls(
        self,
        image_id: str,
    ) -> ImageUrls:
        """Gets the URLs for an image."""

        image_url = f"{self._base_url}/images/{image_id}"
        thumbnail_url = f"{self._base_url}/images/thumbnails/{image_id}.webp"

        return ImageUrls(image_url=image_url, thumbnail_url=thumbnail_url)
