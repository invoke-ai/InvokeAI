from abc import ABC, abstractmethod
from typing import Callable, Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.video_records.video_records_common import (

    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,

)
from invokeai.app.services.videos.videos_common import VideoDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


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
        video_id: str,
        width: int,
        height: int,
        duration: Optional[float] = None,
        frame_rate: Optional[float] = None,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> VideoDTO:
        """Creates a video record and returns its DTO."""
        pass

    @abstractmethod
    def update(
        self,
        video_id: str,
        changes: VideoRecordChanges,
    ) -> VideoDTO:
        """Updates a video record and returns its DTO."""
        pass

    @abstractmethod
    def get_record(self, video_id: str) -> VideoRecord:
        """Gets a video record."""
        pass

    @abstractmethod
    def get_dto(self, video_id: str) -> VideoDTO:
        """Gets a video DTO."""
        pass

    @abstractmethod
    def get_metadata(self, video_id: str) -> Optional[MetadataField]:
        """Gets a video's metadata."""
        pass

    @abstractmethod
    def get_workflow(self, video_id: str) -> Optional[str]:
        """Gets a video's workflow."""
        pass

    @abstractmethod
    def get_graph(self, video_id: str) -> Optional[str]:
        """Gets a video's graph."""
        pass

    @abstractmethod
    def get_path(self, video_id: str, thumbnail: bool = False) -> str:
        """Gets a video's path on disk."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates a video path."""
        pass

    @abstractmethod
    def get_url(self, video_id: str, thumbnail: bool = False) -> str:
        """Gets a video's URL."""
        pass

    @abstractmethod
    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> OffsetPaginatedResults[VideoDTO]:
        """Gets a page of video DTOs."""
        pass

    @abstractmethod
    def delete(self, video_id: str):
        """Deletes a video."""
        pass

    @abstractmethod
    def delete_intermediates(self) -> int:
        """Deletes all intermediate videos and returns the count."""
        pass

    @abstractmethod
    def get_intermediates_count(self) -> int:
        """Gets the count of intermediate videos."""
        pass

    @abstractmethod
    def delete_videos_on_board(self, board_id: str):
        """Deletes all videos on a board."""
        pass

    @abstractmethod
    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> VideoNamesResult:
        """Gets video names with metadata for optimistic updates."""
        pass

