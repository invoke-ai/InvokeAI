from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.video_records.video_records_common import (

    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,

)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class VideoRecordStorageBase(ABC):
    """Low-level service responsible for interfacing with the video record store."""

    @abstractmethod
    def get(self, video_id: str) -> VideoRecord:
        """Gets a video record."""
        pass

    @abstractmethod
    def get_metadata(self, video_id: str) -> Optional[MetadataField]:
        """Gets a video's metadata."""
        pass

    @abstractmethod
    def update(
        self,
        video_id: str,
        changes: VideoRecordChanges,
    ) -> None:
        """Updates a video record."""
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
    ) -> OffsetPaginatedResults[VideoRecord]:
        """Gets a page of video records."""
        pass

    @abstractmethod
    def delete(self, video_id: str) -> None:
        """Deletes a video record."""
        pass

    @abstractmethod
    def delete_many(self, video_ids: list[str]) -> None:
        """Deletes many video records."""
        pass

    @abstractmethod
    def save(
        self,
        video_id: str,
       
        width: int,
        height: int,
        duration: Optional[float] = None,
        frame_rate: Optional[float] = None,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> VideoRecord:
        """Saves a video record."""
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
    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
       
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> VideoNamesResult:
        """Gets video names with metadata for optimistic updates."""
        pass

    @abstractmethod
    def get_intermediates_count(self) -> int:
        """Gets the count of intermediate videos."""
        pass

    @abstractmethod
    def delete_intermediates(self) -> int:
        """Deletes all intermediate videos and returns the count of deleted videos."""
        pass

