from datetime import datetime
from typing import Optional

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ResourceOrigin,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.intermediate_delete import IntermediateDeleteGuard
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection


class ImageRecordStorage(ImageRecordStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def get(self, image_name: str) -> ImageRecord:
        # A database error is deliberately NOT turned into ImageRecordNotFoundException: that made the exception mean
        # "the row is absent, OR the database is locked/corrupt/unreadable". Callers that treat not-found as a benign
        # outcome -- the concurrent-deletion skips in the images and board_images batch routes -- would then swallow a
        # disk I/O error as a routine race and answer 200 with the name in no result list at all.
        record = self._queries.images.get(image_name)
        if record is None:
            raise ImageRecordNotFoundException
        return record

    def set_file_size_bytes(self, image_name: str, file_size_bytes: Optional[int]) -> None:
        self._queries.images.set_file_size(image_name, file_size_bytes)

    def set_file_sizes_bytes(self, sizes: dict[str, int]) -> None:
        self._queries.images.fill_file_sizes(sizes)

    def get_user_id(self, image_name: str) -> Optional[str]:
        return self._queries.images.user_id(image_name)

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        # See get(): a database error must not masquerade as a missing row.
        exists, metadata = self._queries.images.metadata(image_name)
        if not exists:
            raise ImageRecordNotFoundException
        return MetadataFieldValidator.validate_json(metadata) if metadata is not None else None

    def exists(self, image_name: str) -> bool:
        return self._queries.images.exists(image_name)

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> None:
        self._queries.images.update(image_name, changes)

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
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[ImageRecord]:
        images, total = self._queries.images.page(
            offset=offset,
            limit=limit,
            starred_first=starred_first,
            descending=order_dir == SQLiteDirection.Descending,
            image_origin=image_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
            is_admin=is_admin,
        )
        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=total)

    def delete(self, image_name: str) -> None:
        self._queries.images.delete(image_name)

    def delete_many(self, image_names: list[str]) -> None:
        self._queries.images.delete_many(image_names)

    def get_subfolders(self, image_names: list[str]) -> dict[str, str]:
        return self._queries.images.subfolders(image_names)

    def delete_intermediates_by_names(
        self, image_names: list[str], guard: Optional[IntermediateDeleteGuard] = None
    ) -> list[str]:
        """Deletes the named image records, skipping any that are no longer intermediates.

        Returns the names whose records this call actually removed. Names that were already gone, and names whose
        records survive because they are no longer intermediates, are both excluded -- the caller purges the files of
        exactly the returned names and touches nothing else.

        ``guard`` narrows the names on this same transaction; see `IntermediateDeleteGuard`.
        """

        def delete(q: Queries) -> list[str]:
            if guard is None:
                return q.images.delete_intermediates(image_names)
            # Held until the delete commits, so nothing becomes protected between the guard's answer and the delete.
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION)
            return q.images.delete_intermediates(guard(q, image_names))

        return self._queries.run(delete)

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
        project_id: Optional[str] = None,
    ) -> datetime:
        def insert(q: Queries) -> Optional[str]:
            q.images.insert(
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
                user_id=user_id or "system",
                image_subfolder=image_subfolder,
                project_id=project_id,
            )
            # Read back: an image of that name that exists already keeps its row, and its time is the one returned.
            return q.images.created_at(image_name)

        created_at = self._queries.run(insert)
        if created_at is None:
            raise ImageRecordSaveException
        return datetime.fromisoformat(created_at)

    def get_most_recent_image_for_board(self, board_id: str) -> Optional[ImageRecord]:
        return self._queries.images.most_recent_on_board(board_id)

    def get_image_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        image_names, starred_count = self._queries.images.names(
            starred_first=starred_first,
            descending=order_dir == SQLiteDirection.Descending,
            image_origin=image_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            created_from=created_from,
            created_to=created_to,
            user_id=user_id,
            is_admin=is_admin,
        )
        return ImageNamesResult(image_names=image_names, starred_count=starred_count, total_count=len(image_names))

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
        # The day's images that are not intermediates: from its start up to the next day's. A date that is no ISO day
        # matches nothing.
        return self.get_image_names(
            starred_first=starred_first,
            order_dir=order_dir,
            categories=categories,
            is_intermediate=False,
            search_term=search_term,
            created_from=date,
            created_to=date,
            user_id=user_id,
            is_admin=is_admin,
        )
