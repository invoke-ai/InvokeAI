from sqlalchemy import func
from sqlmodel import col, select

from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordDeleteException,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordSaveException,
    BoardVisibility,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.models import BoardTable
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.misc import uuid_string


def _to_record(row: BoardTable) -> BoardRecord:
    """Convert a SQLModel BoardTable row to a BoardRecord pydantic model.

    Must be called while the row is still bound to an active Session.
    """
    try:
        visibility = BoardVisibility(row.board_visibility)
    except ValueError:
        visibility = BoardVisibility.Private

    return BoardRecord(
        board_id=row.board_id,
        board_name=row.board_name,
        user_id=row.user_id,
        cover_image_name=row.cover_image_name,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
        archived=row.archived,
        board_visibility=visibility,
    )


class SqlModelBoardRecordStorage(BoardRecordStorageBase):
    """Board record storage using SQLModel."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def delete(self, board_id: str) -> None:
        with self._db.get_session() as session:
            try:
                board = session.get(BoardTable, board_id)
                if board:
                    session.delete(board)
            except Exception as e:
                raise BoardRecordDeleteException from e

    def save(self, board_name: str, user_id: str) -> BoardRecord:
        board_id = uuid_string()
        board = BoardTable(board_id=board_id, board_name=board_name, user_id=user_id)
        with self._db.get_session() as session:
            try:
                session.add(board)
                session.flush()
                return _to_record(board)
            except Exception as e:
                raise BoardRecordSaveException from e

    def get(self, board_id: str) -> BoardRecord:
        with self._db.get_readonly_session() as session:
            board = session.get(BoardTable, board_id)
            if board is None:
                raise BoardRecordNotFoundException
            return _to_record(board)

    def update(self, board_id: str, changes: BoardChanges) -> BoardRecord:
        with self._db.get_session() as session:
            try:
                board = session.get(BoardTable, board_id)
                if board is None:
                    raise BoardRecordNotFoundException

                if changes.board_name is not None:
                    board.board_name = changes.board_name
                if changes.cover_image_name is not None:
                    board.cover_image_name = changes.cover_image_name
                if changes.archived is not None:
                    board.archived = changes.archived
                if changes.board_visibility is not None:
                    board.board_visibility = changes.board_visibility.value

                session.add(board)
                session.flush()
                return _to_record(board)
            except BoardRecordNotFoundException:
                raise
            except Exception as e:
                raise BoardRecordSaveException from e

    def get_many(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardRecord]:
        with self._db.get_readonly_session() as session:
            # Build filter conditions
            conditions = []

            if not is_admin:
                conditions.append(
                    (col(BoardTable.user_id) == user_id) | (col(BoardTable.board_visibility).in_(["shared", "public"]))
                )

            if not include_archived:
                conditions.append(col(BoardTable.archived) == False)  # noqa: E712

            # Count query
            count_stmt = select(func.count()).select_from(BoardTable)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Data query
            stmt = select(BoardTable)
            for cond in conditions:
                stmt = stmt.where(cond)

            # Apply ordering
            order_col = (
                col(BoardTable.created_at) if order_by == BoardRecordOrderBy.CreatedAt else col(BoardTable.board_name)
            )
            stmt = stmt.order_by(order_col.desc() if direction == SQLiteDirection.Descending else order_col.asc())
            stmt = stmt.offset(offset).limit(limit)

            results = session.exec(stmt).all()
            boards = [_to_record(r) for r in results]

        return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=total)

    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardRecord]:
        with self._db.get_readonly_session() as session:
            stmt = select(BoardTable)

            if not is_admin:
                stmt = stmt.where(
                    (col(BoardTable.user_id) == user_id) | (col(BoardTable.board_visibility).in_(["shared", "public"]))
                )

            if not include_archived:
                stmt = stmt.where(col(BoardTable.archived) == False)  # noqa: E712

            # Apply ordering
            if order_by == BoardRecordOrderBy.Name:
                order_col = col(BoardTable.board_name)
            else:
                order_col = col(BoardTable.created_at)

            stmt = stmt.order_by(order_col.desc() if direction == SQLiteDirection.Descending else order_col.asc())

            results = session.exec(stmt).all()
            boards = [_to_record(r) for r in results]

        return boards
