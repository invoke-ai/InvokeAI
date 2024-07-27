from typing import Optional

from invokeai.app.services.board_images.board_images_base import BoardImagesServiceABC
from invokeai.app.services.invoker import Invoker


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
