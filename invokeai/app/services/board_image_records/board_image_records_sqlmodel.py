from typing import Optional

from sqlalchemy import func
from sqlmodel import col, select

from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    ASSETS_CATEGORIES,
    IMAGE_CATEGORIES,
    ImageCategory,
    ImageRecord,
    deserialize_image_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.models import BoardImageTable, ImageTable
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqlModelBoardImageRecordStorage(BoardImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def add_image_to_board(self, board_id: str, image_name: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(BoardImageTable, image_name)
            if existing is not None:
                existing.board_id = board_id
                session.add(existing)
            else:
                session.add(BoardImageTable(board_id=board_id, image_name=image_name))

    def remove_image_from_board(self, image_name: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(BoardImageTable, image_name)
            if existing is not None:
                session.delete(existing)

    def get_images_for_board(
        self,
        board_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[ImageRecord]:
        with self._db.get_readonly_session() as session:
            # Join board_images with images
            stmt = (
                select(ImageTable)
                .join(BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
                .where(col(BoardImageTable.board_id) == board_id)
                .order_by(col(BoardImageTable.updated_at).desc())
            )
            results = session.exec(stmt).all()
            images = [deserialize_image_record(_image_to_dict(r)) for r in results]

            # Total count of all images
            count_stmt = select(func.count()).select_from(ImageTable)
            count = session.exec(count_stmt).one()

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)

    def get_all_board_image_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
    ) -> list[str]:
        with self._db.get_readonly_session() as session:
            stmt = select(ImageTable.image_name).outerjoin(
                BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name)
            )

            if board_id == "none":
                stmt = stmt.where(col(BoardImageTable.board_id).is_(None))
            else:
                stmt = stmt.where(col(BoardImageTable.board_id) == board_id)

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                stmt = stmt.where(col(ImageTable.image_category).in_(category_strings))

            if is_intermediate is not None:
                stmt = stmt.where(col(ImageTable.is_intermediate) == is_intermediate)

            results = session.exec(stmt).all()
        return list(results)

    def get_board_for_image(self, image_name: str) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            row = session.get(BoardImageTable, image_name)
            if row is None:
                return None
            return row.board_id

    def get_image_count_for_board(self, board_id: str) -> int:
        category_strings = [c.value for c in set(IMAGE_CATEGORIES)]
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(BoardImageTable)
                .join(ImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
                .where(
                    col(ImageTable.is_intermediate) == False,  # noqa: E712
                    col(ImageTable.image_category).in_(category_strings),
                    col(BoardImageTable.board_id) == board_id,
                )
            )
            count = session.exec(stmt).one()
        return count

    def get_asset_count_for_board(self, board_id: str) -> int:
        category_strings = [c.value for c in set(ASSETS_CATEGORIES)]
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(BoardImageTable)
                .join(ImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
                .where(
                    col(ImageTable.is_intermediate) == False,  # noqa: E712
                    col(ImageTable.image_category).in_(category_strings),
                    col(BoardImageTable.board_id) == board_id,
                )
            )
            count = session.exec(stmt).one()
        return count


def _image_to_dict(row: ImageTable) -> dict:
    """Convert an ImageTable row to a dict compatible with deserialize_image_record."""
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
