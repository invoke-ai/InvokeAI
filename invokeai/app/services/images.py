from abc import ABC, abstractmethod
import json
from logging import Logger
from typing import Optional, Union
import uuid
from PIL.Image import Image as PILImageType
from PIL import PngImagePlugin

from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.services.image_record_storage import (
    ImageRecordStorageBase,
)
from invokeai.app.services.models.image_record import (
    ImageRecord,
    ImageDTO,
    image_record_to_dto,
)
from invokeai.app.services.image_file_storage import ImageFileStorageBase
from invokeai.app.services.item_storage import PaginatedResults
from invokeai.app.services.metadata import MetadataServiceBase
from invokeai.app.services.urls import UrlServiceBase
from invokeai.app.util.misc import get_iso_timestamp


class ImageServiceABC(ABC):
    """
    High-level service for image management.

    Provides methods for creating, retrieving, and deleting images.
    """

    @abstractmethod
    def create(
        self,
        image: PILImageType,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[ImageMetadata] = None,
    ) -> ImageDTO:
        """Creates an image, storing the file and its metadata."""
        pass

    @abstractmethod
    def get_pil_image(self, image_type: ImageType, image_name: str) -> PILImageType:
        """Gets an image as a PIL image."""
        pass

    @abstractmethod
    def get_record(self, image_type: ImageType, image_name: str) -> ImageRecord:
        """Gets an image record."""
        pass

    @abstractmethod
    def get_path(self, image_type: ImageType, image_name: str) -> str:
        """Gets an image's path"""
        pass

    @abstractmethod
    def get_url(self, image_type: ImageType, image_name: str, thumbnail: bool = False) -> str:
        """Gets an image's or thumbnail's URL"""
        pass

    @abstractmethod
    def get_dto(self, image_type: ImageType, image_name: str) -> ImageDTO:
        """Gets an image DTO."""
        pass

    @abstractmethod
    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageDTO]:
        """Gets a paginated list of image DTOs."""
        pass

    @abstractmethod
    def delete(self, image_type: ImageType, image_name: str):
        """Deletes an image."""
        pass

    @abstractmethod
    def add_tag(self, image_type: ImageType, image_id: str, tag: str) -> None:
        """Adds a tag to an image."""
        pass

    @abstractmethod
    def remove_tag(self, image_type: ImageType, image_id: str, tag: str) -> None:
        """Removes a tag from an image."""
        pass

    @abstractmethod
    def favorite(self, image_type: ImageType, image_id: str) -> None:
        """Favorites an image."""
        pass

    @abstractmethod
    def unfavorite(self, image_type: ImageType, image_id: str) -> None:
        """Unfavorites an image."""
        pass


class ImageServiceDependencies:
    """Service dependencies for the ImageService."""

    records: ImageRecordStorageBase
    files: ImageFileStorageBase
    metadata: MetadataServiceBase
    urls: UrlServiceBase
    logger: Logger

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
        logger: Logger,
    ):
        self.records = image_record_storage
        self.files = image_file_storage
        self.metadata = metadata
        self.urls = url
        self.logger = logger


class ImageService(ImageServiceABC):
    _services: ImageServiceDependencies

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
        logger: Logger,
    ):
        self._services = ImageServiceDependencies(
            image_record_storage=image_record_storage,
            image_file_storage=image_file_storage,
            metadata=metadata,
            url=url,
            logger=logger,
        )

    def create(
        self,
        image: PILImageType,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[ImageMetadata] = None,
    ) -> ImageDTO:
        image_name = self._create_image_name(
            image_type=image_type,
            image_category=image_category,
            node_id=node_id,
            session_id=session_id,
        )

        timestamp = get_iso_timestamp()

        if metadata is not None:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("invokeai", json.dumps(metadata))
        else:
            pnginfo = None

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            self._services.files.save(
                image_type=image_type,
                image_name=image_name,
                image=image,
                pnginfo=pnginfo,
            )

            self._services.records.save(
                image_name=image_name,
                image_type=image_type,
                image_category=image_category,
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
                created_at=timestamp,
            )

            image_url = self._services.urls.get_image_url(image_type, image_name)
            thumbnail_url = self._services.urls.get_image_url(
                image_type, image_name, True
            )

            return ImageDTO(
                image_name=image_name,
                image_type=image_type,
                image_category=image_category,
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
                created_at=timestamp,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
            )
        except ImageRecordStorageBase.ImageRecordSaveException:
            self._services.logger.error("Failed to save image record")
            raise
        except ImageFileStorageBase.ImageFileSaveException:
            self._services.logger.error("Failed to save image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem saving image record and file")
            raise e

    def get_pil_image(self, image_type: ImageType, image_name: str) -> PILImageType:
        try:
            return self._services.files.get(image_type, image_name)
        except ImageFileStorageBase.ImageFileNotFoundException:
            self._services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_type: ImageType, image_name: str) -> ImageRecord:
        try:
            return self._services.records.get(image_type, image_name)
        except ImageRecordStorageBase.ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image record")
            raise e

    def get_path(
        self, image_type: ImageType, image_name: str, thumbnail: bool = False
    ) -> str:
        try:
            return self._services.files.get_path(image_type, image_name, thumbnail)
        except Exception as e:
            self._services.logger.error("Problem getting image path")
            raise e

    def get_url(
        self, image_type: ImageType, image_name: str, thumbnail: bool = False
    ) -> str:
        try:
            return self._services.urls.get_image_url(image_type, image_name, thumbnail)
        except Exception as e:
            self._services.logger.error("Problem getting image path")
            raise e

    def get_dto(self, image_type: ImageType, image_name: str) -> ImageDTO:
        try:
            image_record = self._services.records.get(image_type, image_name)

            image_dto = image_record_to_dto(
                image_record,
                self._services.urls.get_image_url(image_type, image_name),
                self._services.urls.get_image_url(image_type, image_name, True),
            )

            return image_dto
        except ImageRecordStorageBase.ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image DTO")
            raise e

    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageDTO]:
        try:
            results = self._services.records.get_many(
                image_type,
                image_category,
                page,
                per_page,
            )

            image_dtos = list(
                map(
                    lambda r: image_record_to_dto(
                        r,
                        self._services.urls.get_image_url(image_type, r.image_name),
                        self._services.urls.get_image_url(
                            image_type, r.image_name, True
                        ),
                    ),
                    results.items,
                )
            )

            return PaginatedResults[ImageDTO](
                items=image_dtos,
                page=results.page,
                pages=results.pages,
                per_page=results.per_page,
                total=results.total,
            )
        except Exception as e:
            self._services.logger.error("Problem getting paginated image DTOs")
            raise e

    def delete(self, image_type: ImageType, image_name: str):
        # TODO: Consider using a transaction here to ensure consistency between storage and database
        try:
            self._services.files.delete(image_type, image_name)
            self._services.records.delete(image_type, image_name)
        except ImageRecordStorageBase.ImageRecordDeleteException:
            self._services.logger.error(f"Failed to delete image record")
            raise
        except ImageFileStorageBase.ImageFileDeleteException:
            self._services.logger.error(f"Failed to delete image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem deleting image record and file")
            raise e

    def add_tag(self, image_type: ImageType, image_id: str, tag: str) -> None:
        raise NotImplementedError("The 'add_tag' method is not implemented yet.")

    def remove_tag(self, image_type: ImageType, image_id: str, tag: str) -> None:
        raise NotImplementedError("The 'remove_tag' method is not implemented yet.")

    def favorite(self, image_type: ImageType, image_id: str) -> None:
        raise NotImplementedError("The 'favorite' method is not implemented yet.")

    def unfavorite(self, image_type: ImageType, image_id: str) -> None:
        raise NotImplementedError("The 'unfavorite' method is not implemented yet.")

    def _create_image_name(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Create a unique image name."""
        uuid_str = str(uuid.uuid4())

        if node_id is not None and session_id is not None:
            return f"{image_type.value}_{image_category.value}_{session_id}_{node_id}_{uuid_str}.png"

        return f"{image_type.value}_{image_category.value}_{uuid_str}.png"
