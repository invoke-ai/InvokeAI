from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, TYPE_CHECKING, Union
import uuid
from PIL.Image import Image as PILImageType

from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
    InvalidImageCategoryException,
    InvalidImageTypeException,
)
from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.services.image_record_storage import (
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ImageRecordStorageBase,
)
from invokeai.app.services.models.image_record import (
    ImageRecord,
    ImageDTO,
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
from invokeai.app.services.urls import UrlServiceBase
from invokeai.app.util.misc import get_iso_timestamp

if TYPE_CHECKING:
    from invokeai.app.services.graph import GraphExecutionState


class ImageServiceABC(ABC):
    """High-level service for image management."""

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
    def get_dto(self, image_type: ImageType, image_name: str) -> ImageDTO:
        """Gets an image DTO."""
        pass

    @abstractmethod
    def get_path(self, image_type: ImageType, image_name: str) -> str:
        """Gets an image's path."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates an image's path."""
        pass

    @abstractmethod
    def get_url(
        self, image_type: ImageType, image_name: str, thumbnail: bool = False
    ) -> str:
        """Gets an image's or thumbnail's URL."""
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


class ImageServiceDependencies:
    """Service dependencies for the ImageService."""

    records: ImageRecordStorageBase
    files: ImageFileStorageBase
    metadata: MetadataServiceBase
    urls: UrlServiceBase
    logger: Logger
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
        logger: Logger,
        graph_execution_manager: ItemStorageABC["GraphExecutionState"],
    ):
        self.records = image_record_storage
        self.files = image_file_storage
        self.metadata = metadata
        self.urls = url
        self.logger = logger
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
        graph_execution_manager: ItemStorageABC["GraphExecutionState"],
    ):
        self._services = ImageServiceDependencies(
            image_record_storage=image_record_storage,
            image_file_storage=image_file_storage,
            metadata=metadata,
            url=url,
            logger=logger,
            graph_execution_manager=graph_execution_manager,
        )

    def create(
        self,
        image: PILImageType,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ImageDTO:
        if image_type not in ImageType:
            raise InvalidImageTypeException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self._create_image_name(
            image_type=image_type,
            image_category=image_category,
            node_id=node_id,
            session_id=session_id,
        )

        metadata = self._get_metadata(session_id, node_id)

        (width, height) = image.size

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            created_at = self._services.records.save(
                # Non-nullable fields
                image_name=image_name,
                image_type=image_type,
                image_category=image_category,
                width=width,
                height=height,
                # Nullable fields
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
            )

            self._services.files.save(
                image_type=image_type,
                image_name=image_name,
                image=image,
                metadata=metadata,
            )

            image_url = self._services.urls.get_image_url(image_type, image_name)
            thumbnail_url = self._services.urls.get_image_url(
                image_type, image_name, True
            )

            return ImageDTO(
                # Non-nullable fields
                image_name=image_name,
                image_type=image_type,
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

    def get_pil_image(self, image_type: ImageType, image_name: str) -> PILImageType:
        try:
            return self._services.files.get(image_type, image_name)
        except ImageFileNotFoundException:
            self._services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_type: ImageType, image_name: str) -> ImageRecord:
        try:
            return self._services.records.get(image_type, image_name)
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image record")
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
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image DTO")
            raise e

    def get_path(
        self, image_type: ImageType, image_name: str, thumbnail: bool = False
    ) -> str:
        try:
            return self._services.files.get_path(image_type, image_name, thumbnail)
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
        self, image_type: ImageType, image_name: str, thumbnail: bool = False
    ) -> str:
        try:
            return self._services.urls.get_image_url(image_type, image_name, thumbnail)
        except Exception as e:
            self._services.logger.error("Problem getting image path")
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
        try:
            self._services.files.delete(image_type, image_name)
            self._services.records.delete(image_type, image_name)
        except ImageRecordDeleteException:
            self._services.logger.error(f"Failed to delete image record")
            raise
        except ImageFileDeleteException:
            self._services.logger.error(f"Failed to delete image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem deleting image record and file")
            raise e

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

    def _get_metadata(
        self, session_id: Optional[str] = None, node_id: Optional[str] = None
    ) -> Union[ImageMetadata, None]:
        """Get the metadata for a node."""
        metadata = None

        if node_id is not None and session_id is not None:
            session = self._services.graph_execution_manager.get(session_id)
            metadata = self._services.metadata.create_image_metadata(session, node_id)

        return metadata
