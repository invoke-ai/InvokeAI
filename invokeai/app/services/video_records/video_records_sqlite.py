from typing import Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.video_records.video_records_base import VideoRecordStorageBase
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
)


class SqliteVideoRecordStorage(VideoRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, video_id: str) -> VideoRecord:
        # For now, this is a placeholder that raises NotImplementedError
        # In a real implementation, this would query the videos table
        raise NotImplementedError("Video record storage not yet implemented")

    def get_metadata(self, video_id: str) -> Optional[MetadataField]:
        raise NotImplementedError("Video record storage not yet implemented")

    def update(
        self,
        video_id: str,
        changes: VideoRecordChanges,
    ) -> None:
        raise NotImplementedError("Video record storage not yet implemented")

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> OffsetPaginatedResults[VideoRecord]:
        raise NotImplementedError("Video record storage not yet implemented")

    def delete(self, video_id: str) -> None:
        raise NotImplementedError("Video record storage not yet implemented")

    def delete_many(self, video_ids: list[str]) -> None:
        raise NotImplementedError("Video record storage not yet implemented")

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
        raise NotImplementedError("Video record storage not yet implemented")

    def get_workflow(self, video_id: str) -> Optional[str]:
        raise NotImplementedError("Video record storage not yet implemented")

    def get_graph(self, video_id: str) -> Optional[str]:
        raise NotImplementedError("Video record storage not yet implemented")

    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> VideoNamesResult:
        raise NotImplementedError("Video record storage not yet implemented")

    def get_intermediates_count(self) -> int:
        return 0  # Placeholder implementation

    def delete_intermediates(self) -> int:
        return 0  # Placeholder implementation
