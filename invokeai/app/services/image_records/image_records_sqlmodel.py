from datetime import datetime
from typing import Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ResourceOrigin,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO


class SqlModelImageRecordStorage(ImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def get(self, image_name: str) -> ImageRecord:
        return self._q.images_get(image_name)

    def get_user_id(self, image_name: str) -> Optional[str]:
        return self._q.images_get_user_id(image_name)

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        return self._q.images_get_metadata(image_name)

    def update(self, image_name: str, changes: ImageRecordChanges) -> None:
        self._q.images_update(image_name, changes)

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
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[ImageRecord]:
        return self._q.images_get_many(
            offset=offset,
            limit=limit,
            starred_first=starred_first,
            order_dir=order_dir,
            image_origin=image_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete(self, image_name: str) -> None:
        self._q.images_delete(image_name)

    def delete_many(self, image_names: list[str]) -> None:
        self._q.images_delete_many(image_names)

    def get_intermediates_count(self, user_id: Optional[str] = None) -> int:
        return self._q.images_get_intermediates_count(user_id)

    def delete_intermediates(self) -> list[tuple[str, str]]:
        return self._q.images_delete_intermediates()

    def save(
        self,
        image_name: str,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        width: int,
        height: int,
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[str] = None,
        user_id: Optional[str] = None,
        image_subfolder: str = "",
    ) -> datetime:
        return self._q.images_save(
            image_name=image_name,
            image_origin=image_origin,
            image_category=image_category,
            width=width,
            height=height,
            has_workflow=has_workflow,
            is_intermediate=is_intermediate,
            starred=starred,
            session_id=session_id,
            node_id=node_id,
            metadata=metadata,
            user_id=user_id,
            image_subfolder=image_subfolder,
        )

    def get_most_recent_image_for_board(self, board_id: str) -> Optional[ImageRecord]:
        return self._q.images_get_most_recent_for_board(board_id)

    def get_image_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        return self._q.images_get_names(
            starred_first=starred_first,
            order_dir=order_dir,
            image_origin=image_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )

    def get_image_dates(
        self,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> list[VirtualSubBoardDTO]:
        return self._q.images_get_dates(user_id=user_id, is_admin=is_admin)

    def get_image_names_by_date(
        self,
        date: str,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        categories: Optional[list[ImageCategory]] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        return self._q.images_get_names_by_date(
            date=date,
            starred_first=starred_first,
            order_dir=order_dir,
            categories=categories,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
