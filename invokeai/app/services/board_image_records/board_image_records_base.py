from abc import ABC, abstractmethod
from typing import Optional


class BoardImageRecordStorageBase(ABC):
    """Abstract base class for the one-to-many board-image relationship record storage."""

    @abstractmethod
    def add_image_to_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        """Adds an image to a board."""
        pass

    @abstractmethod
    def remove_image_from_board(
        self,
        image_name: str,
    ) -> None:
        """Removes an image from a board."""
        pass

    @abstractmethod
    def get_all_board_image_names_for_board(
        self,
        board_id: str,
    ) -> list[str]:
        """Gets all board images for a board, as a list of the image names."""
        pass

    @abstractmethod
    def get_board_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        """Gets an image's board id, if it has one."""
        pass

    @abstractmethod
    def get_image_count_for_board(
        self,
        board_id: str,
    ) -> int:
        """Gets the number of images for a board."""
        pass
