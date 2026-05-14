from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.image_records.image_records_common import ImageCategory


class BoardVideoRecordStorageBase(ABC):
    """Abstract base class for the one-to-many board-video relationship record storage."""

    @abstractmethod
    def add_video_to_board(self, board_id: str, video_name: str) -> None:
        """Adds a video to a board."""
        pass

    @abstractmethod
    def remove_video_from_board(self, video_name: str) -> None:
        """Removes a video from a board."""
        pass

    @abstractmethod
    def get_all_board_video_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
    ) -> list[str]:
        """Gets all board videos for a board, as a list of the video names."""
        pass

    @abstractmethod
    def get_board_for_video(self, video_name: str) -> Optional[str]:
        """Gets a video's board id, if it has one."""
        pass

    @abstractmethod
    def get_video_count_for_board(self, board_id: str) -> int:
        """Gets the number of videos for a board."""
        pass
