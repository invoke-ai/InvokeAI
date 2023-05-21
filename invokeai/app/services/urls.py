import os
from abc import ABC, abstractmethod

from invokeai.app.models.image import ImageType
from invokeai.app.util.thumbnails import get_thumbnail_name


class UrlServiceBase(ABC):
    """Responsible for building URLs for resources (eg images or tensors)"""

    @abstractmethod
    def get_image_url(self, image_type: ImageType, image_name: str) -> str:
        """Gets the URL for an image"""
        pass

    @abstractmethod
    def get_thumbnail_url(self, image_type: ImageType, image_name: str) -> str:
        """Gets the URL for an image's thumbnail"""
        pass


class LocalUrlService(UrlServiceBase):
    def __init__(self, base_url: str = "api/v1"):
        self._base_url = base_url

    def get_image_url(self, image_type: ImageType, image_name: str) -> str:
        image_basename = os.path.basename(image_name)
        return f"{self._base_url}/images/{image_type.value}/{image_basename}"

    def get_thumbnail_url(self, image_type: ImageType, image_name: str) -> str:
        thumbnail_basename = get_thumbnail_name(os.path.basename(image_name))
        return f"{self._base_url}/images/{image_type.value}/thumbnails/{thumbnail_basename}"
