from typing import Optional

from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordInboxException,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordProjectNotFoundException,
    BoardRecordProjectUnavailableException,
    BoardRecordSaveException,
    BoardVisibility,
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

    def get_inbox_board_ids(self, board_ids: list[str]) -> set[str]:
        if not board_ids:
            return set()
        # The caller is a board *listing*: an admin's `GET /boards/?all=true` passes every board on the install.
        return set(self._queries.boards.project_ids(board_ids))

    def get_with_inbox_project(self, board_id: str) -> tuple[BoardRecord, Optional[str]]:
        """The board and the project that claims it, in one query: `get_dto` needs both, for every board the
        API returns."""
        found = self._queries.boards.get_with_inbox_project(board_id)
        if found is None:
            raise BoardRecordNotFoundException
        return found

    def save(
        self,
        board_name: str,
        user_id: str,
        project_id: Optional[str] = None,
    ) -> BoardRecord:
        board_id = uuid_string()

        def insert(q: Queries) -> Optional[BoardRecord]:
            if project_id is not None and q.projects.lock(user_id, project_id) is None:
                raise BoardRecordProjectNotFoundException
            q.boards.insert(board_id=board_id, board_name=board_name, user_id=user_id, project_id=project_id)
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
            # A destination project is locked before its board, as in project deletion. A move or
            # creation cannot land in a project whose deletion has already committed.
            destination_exists = True
            if changes.moves_board and changes.project_id is not None:
                current = q.boards.get(board_id)
                if current is None:
                    raise BoardRecordNotFoundException
                destination_exists = q.projects.lock(current.user_id, changes.project_id) is not None
            if q.boards.lock(board_id) is None:
                raise BoardRecordNotFoundException
            found = q.boards.get_with_inbox_project(board_id)
            assert found is not None
            current, inbox_project = found
            if inbox_project is not None and (
                changes.board_name is not None
                or changes.archived is not None
                or changes.board_visibility is not None
                or changes.moves_board
            ):
                raise BoardRecordInboxException
            next_project = changes.project_id if changes.moves_board else current.project_id
            next_visibility = changes.board_visibility or current.board_visibility
            if next_project is not None and (
                next_visibility != BoardVisibility.Private or q.boards.is_shared(board_id)
            ):
                raise BoardRecordProjectUnavailableException
            if not destination_exists:
                raise BoardRecordProjectNotFoundException
            if not q.boards.update(board_id, changes):
                raise BoardRecordInboxException
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
