from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecord, BoardRecordOrderBy
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection


class BoardRecordStorageBase(ABC):
    """Low-level service responsible for interfacing with the board record store."""

    @abstractmethod
    def delete_if_unclaimed(self, board_id: str) -> bool:
        """Delete a board unless it is a project's inbox. Returns whether it was deleted.

        The check and the delete are one statement so that a project claiming the board concurrently
        either commits first and this returns False, or loses and finds the board already gone.
        Callers must not destroy the board's media until this has returned True.
        """
        pass

    @abstractmethod
    def get_inbox_board_ids(self, board_ids: list[str]) -> set[str]:
        """Which of these boards are some project's inbox.

        Bulk because board listings would otherwise issue one lookup per row; chunked internally,
        because a listing can name every board on the install and the query is parameterized per id.
        """
        pass

    @abstractmethod
    def get_with_inbox_project(self, board_id: str) -> tuple[BoardRecord, Optional[str]]:
        """The board record and the id of the project whose inbox it is, if any, in one query."""
        pass

    @abstractmethod
    def save(
        self,
        board_name: str,
        user_id: str,
        project_id: Optional[str] = None,
    ) -> BoardRecord:
        """Saves a board record for a specific user, in one of their projects or in the Library.

        Raises:
            BoardRecordProjectNotFoundException: `project_id` is not one of the user's projects.
        """
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
        """Updates a board record, deciding every rule in one transaction.

        Raises:
            BoardRecordInboxException: The board is a project's inbox and the changes rename,
                archive, publish or move it. Only the cover may change through here.
            BoardRecordProjectNotFoundException: The destination is not one of the owner's projects.
            BoardRecordProjectUnavailableException: The board would be a non-private or explicitly
                shared member of a project.
        """
        pass

    @abstractmethod
    def is_board_shared_with_user(self, board_id: str, user_id: str) -> bool:
        """Checks whether a board has been explicitly shared with a specific user."""
        pass

    @abstractmethod
    def get_shared_user_ids(self, board_id: str) -> list[str]:
        """Gets the ids of the users a board has been explicitly shared with."""
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
    ) -> OffsetPaginatedResults[BoardRecord]:
        """Gets many board records for a specific user, including shared boards. Admin users see all boards."""
        pass

    @abstractmethod
    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardRecord]:
        """Gets all board records for a specific user, including shared boards. Admin users see all boards."""
        pass
