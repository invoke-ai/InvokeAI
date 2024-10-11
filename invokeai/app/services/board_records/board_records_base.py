from abc import ABC, abstractmethod

from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecord, BoardRecordOrderBy
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class BoardRecordStorageBase(ABC):
    """Low-level service responsible for interfacing with the board record store."""

    @abstractmethod
    def delete(self, board_id: str) -> None:
        """Deletes a board record."""
        pass

    @abstractmethod
    def save(
        self,
        board_name: str,
    ) -> BoardRecord:
        """Saves a board record."""
        pass

    @abstractmethod
    def get(
        self,
        board_id: str,
    ) -> BoardRecord:
        """Gets a board record."""
        pass

    @abstractmethod
    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardRecord:
        """Updates a board record."""
        pass

    @abstractmethod
    def get_many(
        self,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardRecord]:
        """Gets many board records."""
        pass

    @abstractmethod
    def get_all(
        self, order_by: BoardRecordOrderBy, direction: SQLiteDirection, include_archived: bool = False
    ) -> list[BoardRecord]:
        """Gets all board records."""
        pass
