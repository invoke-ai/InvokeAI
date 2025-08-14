from typing import Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
)
from invokeai.app.services.videos.videos_base import VideoServiceABC
from invokeai.app.services.videos.videos_common import VideoDTO


class VideoService(VideoServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

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
        # For now, this is a placeholder implementation
        raise NotImplementedError("Video service not yet implemented")

    def update(
        self,
        video_id: str,
        changes: VideoRecordChanges,
    ) -> VideoDTO:
        raise NotImplementedError("Video service not yet implemented")

    def get_record(self, video_id: str) -> VideoRecord:
        raise NotImplementedError("Video service not yet implemented")

    def get_dto(self, video_id: str) -> VideoDTO:
        raise NotImplementedError("Video service not yet implemented")

    def get_metadata(self, video_id: str) -> Optional[MetadataField]:
        raise NotImplementedError("Video service not yet implemented")

    def get_workflow(self, video_id: str) -> Optional[str]:
        raise NotImplementedError("Video service not yet implemented")

    def get_graph(self, video_id: str) -> Optional[str]:
        raise NotImplementedError("Video service not yet implemented")

    def get_path(self, video_id: str, thumbnail: bool = False) -> str:
        raise NotImplementedError("Video service not yet implemented")

    def validate_path(self, path: str) -> bool:
        raise NotImplementedError("Video service not yet implemented")

    def get_url(self, video_id: str, thumbnail: bool = False) -> str:
        raise NotImplementedError("Video service not yet implemented")

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> OffsetPaginatedResults[VideoDTO]:
        # Return empty results for now
        return OffsetPaginatedResults(items=[], offset=offset, limit=limit, total=0, has_more=False)

    def delete(self, video_id: str):
        raise NotImplementedError("Video service not yet implemented")

    def delete_intermediates(self) -> int:
        return 0  # Placeholder

    def get_intermediates_count(self) -> int:
        return 0  # Placeholder

    def delete_videos_on_board(self, board_id: str):
        raise NotImplementedError("Video service not yet implemented")

    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> VideoNamesResult:
        # Return empty results for now
        return VideoNamesResult(video_ids=[])
