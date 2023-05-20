from abc import ABC, abstractmethod
from typing import Optional, Union
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.image import ImageType, ImageCategory
from invokeai.app.services.database.images.models import ImageEntity
from invokeai.app.services.item_storage import PaginatedResults


class ImagesDbServiceBase(ABC):
    """Responsible for interfacing with `images` store."""

    @abstractmethod
    def get(self, id: str) -> Union[ImageEntity, None]:
        """Gets an image from the `images` store."""
        pass

    @abstractmethod
    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageEntity]:
        """Gets a page of images from the `images` store."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Deletes an image from the `images` store."""
        pass

    @abstractmethod
    def set(
        self,
        id: str,
        image_type: ImageType,
        image_category: ImageCategory,
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: Optional[GeneratedImageOrLatentsMetadata | UploadedImageOrLatentsMetadata],
    ) -> None:
        """Sets an image in the `images` store."""
        pass
