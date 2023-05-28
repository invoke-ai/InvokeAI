from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, TYPE_CHECKING, Union
from PIL.Image import Image as PILImageType

from invokeai.app.models.image import (
    ImageCategory,
    ResourceOrigin,
    InvalidImageCategoryException,
    InvalidOriginException,
)
from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.services.image_record_storage import (
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ImageRecordStorageBase,
    OffsetPaginatedResults,
)
from invokeai.app.services.models.image_record import (
    ImageRecord,
    ImageDTO,
    ImageRecordChanges,
    image_record_to_dto,
)
from invokeai.app.services.image_file_storage import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
    ImageFileStorageBase,
)
from invokeai.app.services.item_storage import ItemStorageABC, PaginatedResults
from invokeai.app.services.metadata import MetadataServiceBase
from invokeai.app.services.resource_name import NameServiceBase
from invokeai.app.services.urls import UrlServiceBase

if TYPE_CHECKING:
    from invokeai.app.services.graph import GraphExecutionState


class ImageServiceABC(ABC):
    """High-level service for image management."""

    @abstractmethod
    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        intermediate: bool = False,
    ) -> ImageDTO:
        """Creates an image, storing the file and its metadata."""
        pass

    @abstractmethod
    def update(
        self,
        image_origin: ResourceOrigin,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        """Updates an image."""
        pass

    @abstractmethod
    def get_pil_image(self, image_origin: ResourceOrigin, image_name: str) -> PILImageType:
        """Gets an image as a PIL image."""
        pass

    @abstractmethod
    def get_record(self, image_origin: ResourceOrigin, image_name: str) -> ImageRecord:
        """Gets an image record."""
        pass

    @abstractmethod
    def get_dto(self, image_origin: ResourceOrigin, image_name: str) -> ImageDTO:
        """Gets an image DTO."""
        pass

    @abstractmethod
    def get_path(self, image_origin: ResourceOrigin, image_name: str) -> str:
        """Gets an image's path."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates an image's path."""
        pass

    @abstractmethod
    def get_url(
        self, image_origin: ResourceOrigin, image_name: str, thumbnail: bool = False
    ) -> str:
        """Gets an image's or thumbnail's URL."""
        pass

    @abstractmethod
    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        """Gets a paginated list of image DTOs."""
        pass

    @abstractmethod
    def delete(self, image_origin: ResourceOrigin, image_name: str):
        """Deletes an image."""
        pass


class ImageServiceDependencies:
    """Service dependencies for the ImageService."""

    records: ImageRecordStorageBase
    files: ImageFileStorageBase
    metadata: MetadataServiceBase
    urls: UrlServiceBase
    logger: Logger
    names: NameServiceBase
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
        logger: Logger,
        names: NameServiceBase,
        graph_execution_manager: ItemStorageABC["GraphExecutionState"],
    ):
        self.records = image_record_storage
        self.files = image_file_storage
        self.metadata = metadata
        self.urls = url
        self.logger = logger
        self.names = names
        self.graph_execution_manager = graph_execution_manager


class ImageService(ImageServiceABC):
    _services: ImageServiceDependencies

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
        logger: Logger,
        names: NameServiceBase,
        graph_execution_manager: ItemStorageABC["GraphExecutionState"],
    ):
        self._services = ImageServiceDependencies(
            image_record_storage=image_record_storage,
            image_file_storage=image_file_storage,
            metadata=metadata,
            url=url,
            logger=logger,
            names=names,
            graph_execution_manager=graph_execution_manager,
        )

    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        is_intermediate: bool = False,
    ) -> ImageDTO:
        if image_origin not in ResourceOrigin:
            raise InvalidOriginException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self._services.names.create_image_name()

        metadata = self._get_metadata(session_id, node_id)

        (width, height) = image.size

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            created_at = self._services.records.save(
                # Non-nullable fields
                image_name=image_name,
                image_origin=image_origin,
                image_category=image_category,
                width=width,
                height=height,
                # Meta fields
                is_intermediate=is_intermediate,
                # Nullable fields
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
            )

            self._services.files.save(
                image_origin=image_origin,
                image_name=image_name,
                image=image,
                metadata=metadata,
            )

            image_url = self._services.urls.get_image_url(image_origin, image_name)
            thumbnail_url = self._services.urls.get_image_url(
                image_origin, image_name, True
            )

            return ImageDTO(
                # Non-nullable fields
                image_name=image_name,
                image_origin=image_origin,
                image_category=image_category,
                width=width,
                height=height,
                # Nullable fields
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
                # Meta fields
                created_at=created_at,
                updated_at=created_at,  # this is always the same as the created_at at this time
                deleted_at=None,
                is_intermediate=is_intermediate,
                # Extra non-nullable fields for DTO
                image_url=image_url,
                thumbnail_url=thumbnail_url,
            )
        except ImageRecordSaveException:
            self._services.logger.error("Failed to save image record")
            raise
        except ImageFileSaveException:
            self._services.logger.error("Failed to save image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem saving image record and file")
            raise e

    def update(
        self,
        image_origin: ResourceOrigin,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        try:
            self._services.records.update(image_name, image_origin, changes)
            return self.get_dto(image_origin, image_name)
        except ImageRecordSaveException:
            self._services.logger.error("Failed to update image record")
            raise
        except Exception as e:
            self._services.logger.error("Problem updating image record")
            raise e

    def get_pil_image(self, image_origin: ResourceOrigin, image_name: str) -> PILImageType:
        try:
            return self._services.files.get(image_origin, image_name)
        except ImageFileNotFoundException:
            self._services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_origin: ResourceOrigin, image_name: str) -> ImageRecord:
        try:
            return self._services.records.get(image_origin, image_name)
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image record")
            raise e

    def get_dto(self, image_origin: ResourceOrigin, image_name: str) -> ImageDTO:
        try:
            image_record = self._services.records.get(image_origin, image_name)

            image_dto = image_record_to_dto(
                image_record,
                self._services.urls.get_image_url(image_origin, image_name),
                self._services.urls.get_image_url(image_origin, image_name, True),
            )

            return image_dto
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image DTO")
            raise e

    def get_path(
        self, image_origin: ResourceOrigin, image_name: str, thumbnail: bool = False
    ) -> str:
        try:
            return self._services.files.get_path(image_origin, image_name, thumbnail)
        except Exception as e:
            self._services.logger.error("Problem getting image path")
            raise e

    def validate_path(self, path: str) -> bool:
        try:
            return self._services.files.validate_path(path)
        except Exception as e:
            self._services.logger.error("Problem validating image path")
            raise e

    def get_url(
        self, image_origin: ResourceOrigin, image_name: str, thumbnail: bool = False
    ) -> str:
        try:
            return self._services.urls.get_image_url(image_origin, image_name, thumbnail)
        except Exception as e:
            self._services.logger.error("Problem getting image path")
            raise e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self._services.records.get_many(
                offset,
                limit,
                image_origin,
                categories,
                is_intermediate,
            )

            image_dtos = list(
                map(
                    lambda r: image_record_to_dto(
                        r,
                        self._services.urls.get_image_url(r.image_origin, r.image_name),
                        self._services.urls.get_image_url(
                            r.image_origin, r.image_name, True
                        ),
                    ),
                    results.items,
                )
            )

            return OffsetPaginatedResults[ImageDTO](
                items=image_dtos,
                offset=results.offset,
                limit=results.limit,
                total=results.total,
            )
        except Exception as e:
            self._services.logger.error("Problem getting paginated image DTOs")
            raise e

    def delete(self, image_origin: ResourceOrigin, image_name: str):
        try:
            self._services.files.delete(image_origin, image_name)
            self._services.records.delete(image_origin, image_name)
        except ImageRecordDeleteException:
            self._services.logger.error(f"Failed to delete image record")
            raise
        except ImageFileDeleteException:
            self._services.logger.error(f"Failed to delete image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem deleting image record and file")
            raise e

    def _get_metadata(
        self, session_id: Optional[str] = None, node_id: Optional[str] = None
    ) -> Union[ImageMetadata, None]:
        """Get the metadata for a node."""
        metadata = None

        if node_id is not None and session_id is not None:
            session = self._services.graph_execution_manager.get(session_id)
            metadata = self._services.metadata.create_image_metadata(session, node_id)

        return metadata
