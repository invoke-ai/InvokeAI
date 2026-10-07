from typing import Optional

from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordProjectOwnedException,
    BoardRecordSaveException,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection
from invokeai.app.util.misc import uuid_string


class BoardRecordStorage(BoardRecordStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def delete_if_unclaimed(self, board_id: str) -> bool:
        return self._queries.boards.delete_if_unclaimed(board_id)

    def get_project_ids_for_boards(self, board_ids: list[str]) -> dict[str, str]:
        if not board_ids:
            return {}
        # The caller is a board *listing*: an admin's `GET /boards/?all=true` passes every board on the install.
        return self._queries.boards.project_ids(board_ids)

    def get_with_project_id(self, board_id: str) -> tuple[BoardRecord, Optional[str]]:
        """The board and the project that claims it, in one query: `get_dto` needs both, for every board the
        API returns."""
        found = self._queries.boards.get_with_project_id(board_id)
        if found is None:
            raise BoardRecordNotFoundException
        return found

    def save(
        self,
        board_name: str,
        user_id: str,
    ) -> BoardRecord:
        board_id = uuid_string()

        def insert(q: Queries) -> Optional[BoardRecord]:
            q.boards.insert(board_id=board_id, board_name=board_name, user_id=user_id)
            return q.boards.get(board_id)

        board = self._queries.run(insert)
        if board is None:
            raise BoardRecordSaveException
        return board

    def get(
        self,
        board_id: str,
    ) -> BoardRecord:
        # A database error is deliberately NOT translated into BoardRecordNotFoundException.
        # Translating it made the exception mean "no such board, OR the database is
        # locked/corrupt/unreadable", and callers cannot tell those apart: the batch routes
        # decide board write access off this read once per name, and a not-found answer there is
        # a benign skip. A disk error would therefore drop names out of the response silently,
        # reported neither as moved nor as failed. Mirrors ImageRecordStorage.get.
        board = self._queries.boards.get(board_id)
        if board is None:
            raise BoardRecordNotFoundException
        return board

    def is_board_shared_with_user(self, board_id: str, user_id: str) -> bool:
        return self._queries.boards.is_shared_with(board_id, user_id)

    def get_shared_user_ids(self, board_id: str) -> list[str]:
        return self._queries.boards.shared_user_ids(board_id)

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardRecord:
        def apply(q: Queries) -> Optional[BoardRecord]:
            # One conditional write, not a read followed by a write. A project claim racing this statement
            # either commits first and makes it change nothing, or waits until this update has committed; there
            # is no stale DTO window in which project-owned state can be renamed, archived or published.
            if not q.boards.update(board_id, changes):
                if not q.boards.exists(board_id):
                    raise BoardRecordNotFoundException
                raise BoardRecordProjectOwnedException
            return q.boards.get(board_id)

        board = self._queries.run(apply)
        if board is None:
            raise BoardRecordNotFoundException
        return board

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
        # Admins see every board; other users their own, those shared with them, and shared or public ones.
        boards, total = self._queries.boards.page(
            user_id=user_id,
            is_admin=is_admin,
            order_by=order_by,
            direction=direction,
            offset=offset,
            limit=limit,
            include_archived=include_archived,
        )
        return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=total)

    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardRecord]:
        return self._queries.boards.all(
            user_id=user_id,
            is_admin=is_admin,
            order_by=order_by,
            direction=direction,
            include_archived=include_archived,
        )
