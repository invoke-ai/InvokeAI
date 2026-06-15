from abc import ABC, abstractmethod
from typing import Optional


class BoardCanvasProjectRecordStorageBase(ABC):
    """Abstract base class for the one-to-many board↔canvas-project relationship record storage."""

    @abstractmethod
    def add_project_to_board(self, board_id: str, project_name: str) -> None:
        """Adds a canvas project to a board."""
        pass

    @abstractmethod
    def remove_project_from_board(self, project_name: str) -> None:
        """Removes a canvas project from a board."""
        pass

    @abstractmethod
    def get_all_board_project_names_for_board(
        self,
        board_id: str,
        is_intermediate: Optional[bool] = None,
    ) -> list[str]:
        """Gets all canvas projects for a board, as a list of the project names."""
        pass

    @abstractmethod
    def get_board_for_project(self, project_name: str) -> Optional[str]:
        """Gets a canvas project's board id, if it has one."""
        pass

    @abstractmethod
    def get_project_count_for_board(self, board_id: str) -> int:
        """Gets the number of canvas projects for a board."""
        pass
