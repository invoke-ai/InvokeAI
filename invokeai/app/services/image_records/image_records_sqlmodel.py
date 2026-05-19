from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlmodel import col, select

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ResourceOrigin,
    deserialize_image_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.models import BoardImageTable, ImageTable
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


def _to_dict(row: ImageTable) -> dict:
    return {
        "image_name": row.image_name,
        "image_origin": row.image_origin,
        "image_category": row.image_category,
        "width": row.width,
        "height": row.height,
        "session_id": row.session_id,
        "node_id": row.node_id,
        "metadata": row.metadata_,
        "is_intermediate": row.is_intermediate,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "deleted_at": row.deleted_at,
        "starred": row.starred,
        "has_workflow": row.has_workflow,
    }


class SqlModelImageRecordStorage(ImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, image_name: str) -> ImageRecord:
        with self._db.get_readonly_session() as session:
            row = session.get(ImageTable, image_name)
            if row is None:
                raise ImageRecordNotFoundException
            return deserialize_image_record(_to_dict(row))

    def get_user_id(self, image_name: str) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            row = session.get(ImageTable, image_name)
            if row is None:
                return None
            return row.user_id

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        with self._db.get_readonly_session() as session:
            row = session.get(ImageTable, image_name)
            if row is None:
                raise ImageRecordNotFoundException
            if row.metadata_ is None:
                return None
            return MetadataFieldValidator.validate_json(row.metadata_)

    def update(self, image_name: str, changes: ImageRecordChanges) -> None:
        with self._db.get_session() as session:
            try:
                row = session.get(ImageTable, image_name)
                if row is None:
                    raise ImageRecordNotFoundException

                if changes.image_category is not None:
                    row.image_category = changes.image_category.value
                if changes.session_id is not None:
                    row.session_id = changes.session_id
                if changes.is_intermediate is not None:
                    row.is_intermediate = changes.is_intermediate
                if changes.starred is not None:
                    row.starred = changes.starred

                session.add(row)
            except ImageRecordNotFoundException:
                raise
            except Exception as e:
                raise ImageRecordSaveException from e

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
        with self._db.get_readonly_session() as session:
            # Base query with left join to board_images
            stmt = select(ImageTable).outerjoin(
                BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name)
            )
            count_stmt = (
                select(func.count())
                .select_from(ImageTable)
                .outerjoin(BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
            )

            # Apply filters
            stmt, count_stmt = self._apply_image_filters(
                stmt,
                count_stmt,
                image_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
                user_id,
                is_admin,
            )

            # Count
            total = session.exec(count_stmt).one()

            # Ordering
            if starred_first:
                stmt = stmt.order_by(
                    col(ImageTable.starred).desc(),
                    col(ImageTable.created_at).desc()
                    if order_dir == SQLiteDirection.Descending
                    else col(ImageTable.created_at).asc(),
                )
            else:
                stmt = stmt.order_by(
                    col(ImageTable.created_at).desc()
                    if order_dir == SQLiteDirection.Descending
                    else col(ImageTable.created_at).asc(),
                )

            stmt = stmt.limit(limit).offset(offset)
            results = session.exec(stmt).all()
            images = [deserialize_image_record(_to_dict(r)) for r in results]

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=total)

    def delete(self, image_name: str) -> None:
        with self._db.get_session() as session:
            try:
                row = session.get(ImageTable, image_name)
                if row is not None:
                    session.delete(row)
            except Exception as e:
                raise ImageRecordDeleteException from e

    def delete_many(self, image_names: list[str]) -> None:
        with self._db.get_session() as session:
            try:
                stmt = select(ImageTable).where(col(ImageTable.image_name).in_(image_names))
                rows = session.exec(stmt).all()
                for row in rows:
                    session.delete(row)
            except Exception as e:
                raise ImageRecordDeleteException from e

    def get_intermediates_count(self, user_id: Optional[str] = None) -> int:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(ImageTable)
                .where(
                    col(ImageTable.is_intermediate) == True  # noqa: E712
                )
            )
            if user_id is not None:
                stmt = stmt.where(col(ImageTable.user_id) == user_id)
            count = session.exec(stmt).one()
        return count

    def delete_intermediates(self) -> list[str]:
        with self._db.get_session() as session:
            try:
                stmt = select(ImageTable).where(col(ImageTable.is_intermediate) == True)  # noqa: E712
                rows = session.exec(stmt).all()
                names = [r.image_name for r in rows]
                for row in rows:
                    session.delete(row)
            except Exception as e:
                raise ImageRecordDeleteException from e
        return names

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
    ) -> datetime:
        row = ImageTable(
            image_name=image_name,
            image_origin=image_origin.value,
            image_category=image_category.value,
            width=width,
            height=height,
            session_id=session_id,
            node_id=node_id,
            metadata_=metadata,
            is_intermediate=is_intermediate or False,
            starred=starred or False,
            has_workflow=has_workflow,
            user_id=user_id or "system",
        )
        with self._db.get_session() as session:
            try:
                session.add(row)
                session.flush()
                # With expire_on_commit=False, row.created_at is still accessible
                return (
                    row.created_at
                    if isinstance(row.created_at, datetime)
                    else datetime.fromisoformat(str(row.created_at))
                )
            except Exception as e:
                raise ImageRecordSaveException from e

    def get_most_recent_image_for_board(self, board_id: str) -> Optional[ImageRecord]:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(ImageTable)
                .join(BoardImageTable, col(ImageTable.image_name) == col(BoardImageTable.image_name))
                .where(
                    col(BoardImageTable.board_id) == board_id,
                    col(ImageTable.is_intermediate) == False,  # noqa: E712
                )
                .order_by(col(ImageTable.starred).desc(), col(ImageTable.created_at).desc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            if row is None:
                return None
            return deserialize_image_record(_to_dict(row))

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
        with self._db.get_readonly_session() as session:
            # Base query
            stmt = select(ImageTable.image_name).outerjoin(
                BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name)
            )

            # Dummy count stmt for filter reuse (we won't use it here)
            count_stmt = (
                select(func.count())
                .select_from(ImageTable)
                .outerjoin(BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
            )

            stmt, count_stmt = self._apply_image_filters(
                stmt,
                count_stmt,
                image_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
                user_id,
                is_admin,
            )

            # Starred count
            starred_count = 0
            if starred_first:
                starred_stmt = count_stmt.where(col(ImageTable.starred) == True)  # noqa: E712
                starred_count = session.exec(starred_stmt).one()

            # Ordering
            if starred_first:
                stmt = stmt.order_by(
                    col(ImageTable.starred).desc(),
                    col(ImageTable.created_at).desc()
                    if order_dir == SQLiteDirection.Descending
                    else col(ImageTable.created_at).asc(),
                )
            else:
                stmt = stmt.order_by(
                    col(ImageTable.created_at).desc()
                    if order_dir == SQLiteDirection.Descending
                    else col(ImageTable.created_at).asc(),
                )

            results = session.exec(stmt).all()

        return ImageNamesResult(
            image_names=list(results),
            starred_count=starred_count,
            total_count=len(results),
        )

    @staticmethod
    def _apply_image_filters(
        stmt, count_stmt, image_origin, categories, is_intermediate, board_id, search_term, user_id, is_admin
    ):
        """Apply common filters to both data and count queries."""
        if image_origin is not None:
            cond = col(ImageTable.image_origin) == image_origin.value
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if categories is not None:
            category_strings = [c.value for c in set(categories)]
            cond = col(ImageTable.image_category).in_(category_strings)
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if is_intermediate is not None:
            cond = col(ImageTable.is_intermediate) == is_intermediate
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if board_id == "none":
            cond = col(BoardImageTable.board_id).is_(None)
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)
            if user_id is not None and not is_admin:
                user_cond = col(ImageTable.user_id) == user_id
                stmt = stmt.where(user_cond)
                count_stmt = count_stmt.where(user_cond)
        elif board_id is not None:
            cond = col(BoardImageTable.board_id) == board_id
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        if search_term:
            term = f"%{search_term.lower()}%"
            cond = col(ImageTable.metadata_).like(term) | col(ImageTable.created_at).like(term)
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        return stmt, count_stmt
