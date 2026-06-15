from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
)


class VideoRecordStorageBase(ABC):
    """Low-level service responsible for interfacing with the video record store."""

    @abstractmethod
    def get(self, video_name: str) -> VideoRecord:
        """Gets a video record."""
        pass

    @abstractmethod
    def get_metadata(self, video_name: str) -> Optional[MetadataField]:
        """Gets a video's metadata."""
        pass

    @abstractmethod
    def update(self, video_name: str, changes: VideoRecordChanges) -> None:
        """Updates a video record."""
        pass

    @abstractmethod
    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        video_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[VideoRecord]:
        """Gets a page of video records."""
        pass

    @abstractmethod
    def delete(self, video_name: str) -> None:
        """Deletes a video record."""
        pass

    @abstractmethod
    def delete_many(self, video_names: list[str]) -> None:
        """Deletes many video records."""
        pass

    @abstractmethod
    def save(
        self,
        video_name: str,
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        width: int,
        height: int,
        duration: float,
        fps: Optional[float],
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[str] = None,
        user_id: Optional[str] = None,
        video_subfolder: str = "",
    ) -> datetime:
        """Saves a video record."""
        pass

    @abstractmethod
    def get_user_id(self, video_name: str) -> Optional[str]:
        """Gets the user_id of the video owner. Returns None if video not found."""
        pass

    @abstractmethod
    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        video_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> VideoNamesResult:
        """Gets ordered list of video names with metadata for optimistic updates."""
        pass
