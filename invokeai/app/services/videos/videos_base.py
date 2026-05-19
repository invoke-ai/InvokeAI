from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
)
from invokeai.app.services.videos.videos_common import VideoDTO


class VideoServiceABC(ABC):
    """High-level service for video management."""

    _on_changed_callbacks: list[Callable[[VideoDTO], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = []
        self._on_deleted_callbacks = []

    def on_changed(self, on_changed: Callable[[VideoDTO], None]) -> None:
        """Register a callback for when a video is changed."""
        self._on_changed_callbacks.append(on_changed)

    def on_deleted(self, on_deleted: Callable[[str], None]) -> None:
        """Register a callback for when a video is deleted."""
        self._on_deleted_callbacks.append(on_deleted)

    def _on_changed(self, item: VideoDTO) -> None:
        for callback in self._on_changed_callbacks:
            callback(item)

    def _on_deleted(self, item_id: str) -> None:
        for callback in self._on_deleted_callbacks:
            callback(item_id)

    @abstractmethod
    def create(
        self,
        source_path: Path,
        width: int,
        height: int,
        duration: float,
        fps: Optional[float],
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> VideoDTO:
        """Creates a video by moving/copying the file at `source_path` into storage and recording it."""
        pass

    @abstractmethod
    def update(self, video_name: str, changes: VideoRecordChanges) -> VideoDTO:
        """Updates a video."""
        pass

    @abstractmethod
    def get_record(self, video_name: str) -> VideoRecord:
        """Gets a video record."""
        pass

    @abstractmethod
    def get_dto(self, video_name: str) -> VideoDTO:
        """Gets a video DTO."""
        pass

    @abstractmethod
    def get_metadata(self, video_name: str) -> Optional[MetadataField]:
        """Gets a video's metadata."""
        pass

    @abstractmethod
    def get_workflow(self, video_name: str) -> Optional[str]:
        """Gets a video's workflow."""
        pass

    @abstractmethod
    def get_graph(self, video_name: str) -> Optional[str]:
        """Gets a video's graph."""
        pass

    @abstractmethod
    def get_path(self, video_name: str, thumbnail: bool = False) -> str:
        """Gets a video's on-disk path."""
        pass

    @abstractmethod
    def get_url(self, video_name: str, thumbnail: bool = False) -> str:
        """Gets a video's URL."""
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
    ) -> OffsetPaginatedResults[VideoDTO]:
        """Gets a paginated list of video DTOs."""
        pass

    @abstractmethod
    def delete(self, video_name: str) -> None:
        """Deletes a video."""
        pass

    @abstractmethod
    def delete_videos_on_board(self, board_id: str, user_id: Optional[str] = None) -> None:
        """Deletes all videos on a board.

        When ``user_id`` is provided, only videos owned by that user are deleted (other users'
        contributions to a public/shared board are preserved). Pass ``None`` for the admin
        path to delete every video on the board regardless of uploader.
        """
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
        """Gets ordered list of video names."""
        pass
