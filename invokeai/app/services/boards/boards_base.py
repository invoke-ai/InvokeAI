from abc import ABC, abstractmethod

from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecordOrderBy
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class BoardServiceABC(ABC):
    """High-level service for board management."""

    @abstractmethod
    def create(
        self,
        board_name: str,
        user_id: str,
    ) -> BoardDTO:
        """Creates a board for a specific user."""
        pass

    @abstractmethod
    def get_dto(
        self,
        board_id: str,
    ) -> BoardDTO:
        """Gets a board."""
        pass

    @abstractmethod
    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        """Updates a board."""
        pass

    @abstractmethod
    def delete_if_unclaimed(
        self,
        board_id: str,
    ) -> bool:
        """Delete a board only if no project owns it. Returns whether it was deleted.

        The only deletion there is. An unconditional `delete` used to sit beside this one with no
        callers, which made "a project's board cannot be deleted out from under it" true by
        coincidence rather than by construction — the next caller to reach for the obvious name
        would have silently given it up.
        """
        pass

    @abstractmethod
    def get_many(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardDTO]:
        """Gets many boards for a specific user, including shared boards. Admin users see all boards."""
        pass

    @abstractmethod
    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardDTO]:
        """Gets all boards for a specific user, including shared boards. Admin users see all boards."""
        pass
