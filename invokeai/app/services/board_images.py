from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.board_record_storage import BoardRecord
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.models.board_record import BoardDTO


class BoardImagesServiceABC(ABC):
    """High-level service for board-image relationship management."""

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


class BoardImagesService(BoardImagesServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def add_image_to_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        self.__invoker.services.board_image_records.add_image_to_board(board_id, image_name)

    def remove_image_from_board(
        self,
        image_name: str,
    ) -> None:
        self.__invoker.services.board_image_records.remove_image_from_board(image_name)

    def get_all_board_image_names_for_board(
        self,
        board_id: str,
    ) -> list[str]:
        return self.__invoker.services.board_image_records.get_all_board_image_names_for_board(board_id)

    def get_board_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        board_id = self.__invoker.services.board_image_records.get_board_for_image(image_name)
        return board_id


def board_record_to_dto(board_record: BoardRecord, cover_image_name: Optional[str], image_count: int) -> BoardDTO:
    """Converts a board record to a board DTO."""
    return BoardDTO(
        **board_record.dict(exclude={"cover_image_name"}),
        cover_image_name=cover_image_name,
        image_count=image_count,
    )
