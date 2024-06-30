from abc import ABC, abstractmethod
from typing import Callable, Optional

from PIL.Image import Image as PILImageType

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecord,
    ImageRecordChanges,
    ResourceOrigin,
)
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class ImageServiceABC(ABC):
    """High-level service for image management."""

    _on_changed_callbacks: list[Callable[[ImageDTO], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = []
        self._on_deleted_callbacks = []

    def on_changed(self, on_changed: Callable[[ImageDTO], None]) -> None:
        """Register a callback for when an image is changed"""
        self._on_changed_callbacks.append(on_changed)

    def on_deleted(self, on_deleted: Callable[[str], None]) -> None:
        """Register a callback for when an image is deleted"""
        self._on_deleted_callbacks.append(on_deleted)

    def _on_changed(self, item: ImageDTO) -> None:
        for callback in self._on_changed_callbacks:
            callback(item)

    def _on_deleted(self, item_id: str) -> None:
        for callback in self._on_deleted_callbacks:
            callback(item_id)

    @abstractmethod
    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
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
    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        """Gets an image's metadata."""
        pass

    @abstractmethod
    def get_workflow(self, image_name: str) -> Optional[str]:
        """Gets an image's workflow."""
        pass

    @abstractmethod
    def get_graph(self, image_name: str) -> Optional[str]:
        """Gets an image's workflow."""
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
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
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
    def get_intermediates_count(self) -> int:
        """Gets the number of intermediate images."""
        pass

    @abstractmethod
    def delete_images_on_board(self, board_id: str):
        """Deletes all images on a board."""
        pass
