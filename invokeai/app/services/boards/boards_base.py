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
    ) -> BoardDTO:
        """Creates a board."""
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
    def delete(
        self,
        board_id: str,
    ) -> None:
        """Deletes a board."""
        pass

    @abstractmethod
    def get_many(
        self,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardDTO]:
        """Gets many boards."""
        pass

    @abstractmethod
    def get_all(
        self, order_by: BoardRecordOrderBy, direction: SQLiteDirection, include_archived: bool = False
    ) -> list[BoardDTO]:
        """Gets all boards."""
        pass
