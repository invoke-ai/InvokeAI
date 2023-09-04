from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, Optional

from PIL.Image import Image as PILImageType

from invokeai.app.invocations.metadata import ImageMetadata
from invokeai.app.models.image import (
    ImageCategory,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from invokeai.app.services.board_image_record_storage import BoardImageRecordStorageBase
from invokeai.app.services.image_file_storage import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
    ImageFileStorageBase,
)
from invokeai.app.services.image_record_storage import (
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ImageRecordStorageBase,
    OffsetPaginatedResults,
)
from invokeai.app.services.item_storage import ItemStorageABC
from invokeai.app.services.models.image_record import (
    ImageDTO,
    ImageRecord,
    ImageRecordChanges,
    image_record_to_dto,
)
from invokeai.app.services.resource_name import NameServiceBase
from invokeai.app.services.urls import UrlServiceBase
from invokeai.app.util.metadata import get_metadata_graph_from_raw_session

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
        board_id: Optional[str] = None,
        is_intermediate: bool = False,
        metadata: Optional[dict] = None,
        workflow: Optional[str] = None,
    ) -> ImageDTO:
        """Creates an image, storing the file and its metadata."""
        pass

    @abstractmethod
    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        """Updates an image."""
        pass

    @abstractmethod
    def get_pil_image(self, image_name: str) -> PILImageType:
        """Gets an image as a PIL image."""
        pass

    @abstractmethod
    def get_record(self, image_name: str) -> ImageRecord:
        """Gets an image record."""
        pass

    @abstractmethod
    def get_dto(self, image_name: str) -> ImageDTO:
        """Gets an image DTO."""
        pass

    @abstractmethod
    def get_metadata(self, image_name: str) -> ImageMetadata:
        """Gets an image's metadata."""
        pass

    @abstractmethod
    def get_path(self, image_name: str, thumbnail: bool = False) -> str:
        """Gets an image's path."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates an image's path."""
        pass

    @abstractmethod
    def get_url(self, image_name: str, thumbnail: bool = False) -> str:
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
        board_id: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        """Gets a paginated list of image DTOs."""
        pass

    @abstractmethod
    def delete(self, image_name: str):
        """Deletes an image."""
        pass

    @abstractmethod
    def delete_intermediates(self) -> int:
        """Deletes all intermediate images."""
        pass

    @abstractmethod
    def delete_images_on_board(self, board_id: str):
        """Deletes all images on a board."""
        pass


class ImageServiceDependencies:
    """Service dependencies for the ImageService."""

    image_records: ImageRecordStorageBase
    image_files: ImageFileStorageBase
    board_image_records: BoardImageRecordStorageBase
    urls: UrlServiceBase
    logger: Logger
    names: NameServiceBase
    graph_execution_manager: ItemStorageABC["GraphExecutionState"]

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        board_image_record_storage: BoardImageRecordStorageBase,
        url: UrlServiceBase,
        logger: Logger,
        names: NameServiceBase,
        graph_execution_manager: ItemStorageABC["GraphExecutionState"],
    ):
        self.image_records = image_record_storage
        self.image_files = image_file_storage
        self.board_image_records = board_image_record_storage
        self.urls = url
        self.logger = logger
        self.names = names
        self.graph_execution_manager = graph_execution_manager


class ImageService(ImageServiceABC):
    _services: ImageServiceDependencies

    def __init__(self, services: ImageServiceDependencies):
        self._services = services

    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: bool = False,
        metadata: Optional[dict] = None,
        workflow: Optional[str] = None,
    ) -> ImageDTO:
        if image_origin not in ResourceOrigin:
            raise InvalidOriginException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self._services.names.create_image_name()

        # TODO: Do we want to store the graph in the image at all? I don't think so...
        # graph = None
        # if session_id is not None:
        #     session_raw = self._services.graph_execution_manager.get_raw(session_id)
        #     if session_raw is not None:
        #         try:
        #             graph = get_metadata_graph_from_raw_session(session_raw)
        #         except Exception as e:
        #             self._services.logger.warn(f"Failed to parse session graph: {e}")
        #             graph = None

        (width, height) = image.size

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            self._services.image_records.save(
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
                metadata=metadata,
                session_id=session_id,
            )
            if board_id is not None:
                self._services.board_image_records.add_image_to_board(board_id=board_id, image_name=image_name)
            self._services.image_files.save(image_name=image_name, image=image, metadata=metadata, workflow=workflow)
            image_dto = self.get_dto(image_name)

            return image_dto
        except ImageRecordSaveException:
            self._services.logger.error("Failed to save image record")
            raise
        except ImageFileSaveException:
            self._services.logger.error("Failed to save image file")
            raise
        except Exception as e:
            self._services.logger.error(f"Problem saving image record and file: {str(e)}")
            raise e

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        try:
            self._services.image_records.update(image_name, changes)
            return self.get_dto(image_name)
        except ImageRecordSaveException:
            self._services.logger.error("Failed to update image record")
            raise
        except Exception as e:
            self._services.logger.error("Problem updating image record")
            raise e

    def get_pil_image(self, image_name: str) -> PILImageType:
        try:
            return self._services.image_files.get(image_name)
        except ImageFileNotFoundException:
            self._services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_name: str) -> ImageRecord:
        try:
            return self._services.image_records.get(image_name)
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image record")
            raise e

    def get_dto(self, image_name: str) -> ImageDTO:
        try:
            image_record = self._services.image_records.get(image_name)

            image_dto = image_record_to_dto(
                image_record,
                self._services.urls.get_image_url(image_name),
                self._services.urls.get_image_url(image_name, True),
                self._services.board_image_records.get_board_for_image(image_name),
            )

            return image_dto
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image DTO")
            raise e

    def get_metadata(self, image_name: str) -> Optional[ImageMetadata]:
        try:
            image_record = self._services.image_records.get(image_name)
            metadata = self._services.image_records.get_metadata(image_name)

            if not image_record.session_id:
                return ImageMetadata(metadata=metadata)

            session_raw = self._services.graph_execution_manager.get_raw(image_record.session_id)
            graph = None

            if session_raw:
                try:
                    graph = get_metadata_graph_from_raw_session(session_raw)
                except Exception as e:
                    self._services.logger.warn(f"Failed to parse session graph: {e}")
                    graph = None

            return ImageMetadata(graph=graph, metadata=metadata)
        except ImageRecordNotFoundException:
            self._services.logger.error("Image record not found")
            raise
        except Exception as e:
            self._services.logger.error("Problem getting image DTO")
            raise e

    def get_path(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return self._services.image_files.get_path(image_name, thumbnail)
        except Exception as e:
            self._services.logger.error("Problem getting image path")
            raise e

    def validate_path(self, path: str) -> bool:
        try:
            return self._services.image_files.validate_path(path)
        except Exception as e:
            self._services.logger.error("Problem validating image path")
            raise e

    def get_url(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return self._services.urls.get_image_url(image_name, thumbnail)
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
        board_id: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self._services.image_records.get_many(
                offset,
                limit,
                image_origin,
                categories,
                is_intermediate,
                board_id,
            )

            image_dtos = list(
                map(
                    lambda r: image_record_to_dto(
                        r,
                        self._services.urls.get_image_url(r.image_name),
                        self._services.urls.get_image_url(r.image_name, True),
                        self._services.board_image_records.get_board_for_image(r.image_name),
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

    def delete(self, image_name: str):
        try:
            self._services.image_files.delete(image_name)
            self._services.image_records.delete(image_name)
        except ImageRecordDeleteException:
            self._services.logger.error("Failed to delete image record")
            raise
        except ImageFileDeleteException:
            self._services.logger.error("Failed to delete image file")
            raise
        except Exception as e:
            self._services.logger.error("Problem deleting image record and file")
            raise e

    def delete_images_on_board(self, board_id: str):
        try:
            image_names = self._services.board_image_records.get_all_board_image_names_for_board(board_id)
            for image_name in image_names:
                self._services.image_files.delete(image_name)
            self._services.image_records.delete_many(image_names)
        except ImageRecordDeleteException:
            self._services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self._services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self._services.logger.error("Problem deleting image records and files")
            raise e

    def delete_intermediates(self) -> int:
        try:
            image_names = self._services.image_records.delete_intermediates()
            count = len(image_names)
            for image_name in image_names:
                self._services.image_files.delete(image_name)
            return count
        except ImageRecordDeleteException:
            self._services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self._services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self._services.logger.error("Problem deleting image records and files")
            raise e
