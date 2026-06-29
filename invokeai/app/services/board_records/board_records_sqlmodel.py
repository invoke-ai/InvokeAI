from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordOrderBy,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqlModelBoardRecordStorage(BoardRecordStorageBase):
    """Board record storage using SQLModel."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def delete(self, board_id: str) -> None:
        self._q.boards_delete(board_id)

    def save(self, board_name: str, user_id: str) -> BoardRecord:
        return self._q.boards_save(board_name, user_id)

    def get(self, board_id: str) -> BoardRecord:
        return self._q.boards_get(board_id)

    def update(self, board_id: str, changes: BoardChanges) -> BoardRecord:
        return self._q.boards_update(board_id, changes)

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
        return self._q.boards_get_many(
            user_id=user_id,
            is_admin=is_admin,
            order_by=order_by,
            direction=direction,
            offset=offset,
            limit=limit,
            include_archived=include_archived,
        )

    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardRecord]:
        return self._q.boards_get_all(
            user_id=user_id,
            is_admin=is_admin,
            order_by=order_by,
            direction=direction,
            include_archived=include_archived,
        )
