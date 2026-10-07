from datetime import datetime
from typing import Optional

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.intermediate_delete import IntermediateDeleteGuard
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection
from invokeai.app.services.video_records.video_records_base import VideoRecordStorageBase
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
    VideoRecordNotFoundException,
    VideoRecordSaveException,
)


class VideoRecordStorage(VideoRecordStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def get(self, video_name: str) -> VideoRecord:
        # A database error is deliberately NOT turned into VideoRecordNotFoundException: it would make the exception
        # mean "the row is absent, OR the database is unreadable". `_assert_video_read_access` answers 404 on a
        # not-found, and the clients drop their references to a video on that 404 -- so a locked database would
        # silently clear the user's workflow fields.
        record = self._queries.videos.get(video_name)
        if record is None:
            raise VideoRecordNotFoundException
        return record

    def get_subfolders(self, video_names: list[str]) -> dict[str, str]:
        return self._queries.videos.subfolders(video_names)

    def delete_intermediates_by_names(
        self, video_names: list[str], guard: Optional[IntermediateDeleteGuard] = None
    ) -> list[str]:
        def delete(q: Queries) -> list[str]:
            if guard is None:
                return q.videos.delete_intermediates(video_names)
            # Held until the delete commits, so nothing becomes protected between the guard's answer and the delete.
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION)
            return q.videos.delete_intermediates(guard(q, video_names))

        return self._queries.run(delete)

    def set_file_size_bytes(self, video_name: str, file_size_bytes: Optional[int]) -> None:
        self._queries.videos.set_file_size(video_name, file_size_bytes)

    def set_file_sizes_bytes(self, sizes: dict[str, int]) -> None:
        self._queries.videos.fill_file_sizes(sizes)

    def get_user_id(self, video_name: str) -> Optional[str]:
        return self._queries.videos.user_id(video_name)

    def get_most_recent_video_for_board(self, board_id: str) -> Optional[VideoRecord]:
        return self._queries.videos.most_recent_on_board(board_id)

    def exists(self, video_name: str) -> bool:
        return self._queries.videos.exists(video_name)

    def get_metadata(self, video_name: str) -> Optional[MetadataField]:
        # See get(): a database error must not masquerade as a missing row.
        exists, metadata = self._queries.videos.metadata(video_name)
        if not exists:
            raise VideoRecordNotFoundException
        return MetadataFieldValidator.validate_json(metadata) if metadata is not None else None

    def update(self, video_name: str, changes: VideoRecordChanges) -> None:
        self._queries.videos.update(video_name, changes)

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
        videos, total = self._queries.videos.page(
            offset=offset,
            limit=limit,
            starred_first=starred_first,
            descending=order_dir == SQLiteDirection.Descending,
            video_origin=video_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
        return OffsetPaginatedResults(items=videos, offset=offset, limit=limit, total=total)

    def delete(self, video_name: str) -> None:
        self._queries.videos.delete(video_name)

    def delete_many(self, video_names: list[str]) -> None:
        self._queries.videos.delete_many(video_names)

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
        project_id: Optional[str] = None,
    ) -> datetime:
        def insert(q: Queries) -> Optional[str]:
            q.videos.insert(
                video_name=video_name,
                video_origin=video_origin,
                video_category=video_category,
                width=width,
                height=height,
                duration=duration,
                fps=fps,
                has_workflow=has_workflow,
                is_intermediate=is_intermediate,
                starred=starred,
                session_id=session_id,
                node_id=node_id,
                metadata=metadata,
                user_id=user_id or "system",
                video_subfolder=video_subfolder,
                project_id=project_id,
            )
            # Read back: a video of that name that exists already keeps its row, and its time is the one returned.
            return q.videos.created_at(video_name)

        created_at = self._queries.run(insert)
        if created_at is None:
            raise VideoRecordSaveException
        return datetime.fromisoformat(created_at)

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
        video_names, starred_count = self._queries.videos.names(
            starred_first=starred_first,
            descending=order_dir == SQLiteDirection.Descending,
            video_origin=video_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
        return VideoNamesResult(video_names=video_names, starred_count=starred_count, total_count=len(video_names))
