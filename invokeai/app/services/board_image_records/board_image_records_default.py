from typing import Optional

from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.database.database import Database


class BoardImageRecordStorage(BoardImageRecordStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def add_image_to_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        self._queries.board_images.add(board_id, image_name)

    def remove_image_from_board(
        self,
        image_name: str,
        board_id: str,
    ) -> int:
        # Scoped to the board the caller was authorized against, not just the image. The
        # routes read the image's board, authorize against *that* board, and only then
        # remove; an unscoped DELETE would follow the image if it were moved in between,
        # applying a decision taken about one board to a different one. Zero rows means the
        # image left this board between the caller's read and this write.
        return int(self._queries.board_images.remove(image_name, board_id))

    def get_all_board_image_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
        user_id: Optional[str] = None,
    ) -> list[str]:
        # `board_id` "none" means the images on no board; admins pass user_id=None to see every account's.
        return self._queries.board_images.image_names(
            None if board_id == "none" else board_id,
            categories=categories,
            is_intermediate=is_intermediate,
            user_id=user_id,
        )

    def get_board_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        return self._queries.board_images.board_of(image_name)

    def get_counts_for_board(self, board_id: str) -> tuple[int, int]:
        return self._queries.board_images.counts(board_id)
