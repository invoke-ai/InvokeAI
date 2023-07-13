import os
from abc import ABC, abstractmethod


class UrlServiceBase(ABC):
    """Responsible for building URLs for resources."""

    @abstractmethod
    def get_image_url(self, image_name: str, thumbnail: bool = False) -> str:
        """Gets the URL for an image or thumbnail."""
        pass


class LocalUrlService(UrlServiceBase):
    def __init__(self, base_url: str = "api/v1"):
        self._base_url = base_url

    def get_image_url(self, image_name: str, thumbnail: bool = False) -> str:
        image_basename = os.path.basename(image_name)

        # These paths are determined by the routes in invokeai/app/api/routers/images.py
        if thumbnail:
            return f"{self._base_url}/images/{image_basename}/thumbnail"

        return f"{self._base_url}/images/{image_basename}/full"
