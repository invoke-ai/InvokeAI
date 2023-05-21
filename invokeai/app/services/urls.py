import os
from abc import ABC, abstractmethod

from invokeai.app.models.image import ImageType
from invokeai.app.util.thumbnails import get_thumbnail_name


class URLServiceBase(ABC):
    """Responsible for building URLs for resources (eg images or tensors)"""

    @abstractmethod
    def get_image_url(
        self, image_type: ImageType, image_id: str, thumbnail: bool = False
    ) -> str:
        """Gets the URL for an image"""
        pass


class LocalURLService(URLServiceBase):
    def __init__(self, base_url: str = "api/v1"):
        self._base_url = base_url

    def get_image_url(
        self, image_type: ImageType, image_id: str, thumbnail: bool = False
    ) -> str:
        image_basename = os.path.basename(image_id)

        if thumbnail:
            thumbnail_basename = get_thumbnail_name(image_basename)
            url = f"{self._base_url}/images/{image_type.value}/thumbnails/{thumbnail_basename}"
        else:
            url = f"{self._base_url}/images/{image_type.value}/{image_basename}"

        return url
