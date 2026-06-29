from typing import Optional

from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecord
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqlModelBoardImageRecordStorage(BoardImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def add_image_to_board(self, board_id: str, image_name: str) -> None:
        self._q.board_images_add(board_id, image_name)

    def remove_image_from_board(self, image_name: str) -> None:
        self._q.board_images_remove(image_name)

    def get_images_for_board(
        self,
        board_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[ImageRecord]:
        return self._q.board_images_get_images_for_board(board_id, offset=offset, limit=limit)

    def get_all_board_image_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
    ) -> list[str]:
        return self._q.board_images_get_all_image_names_for_board(board_id, categories, is_intermediate)

    def get_board_for_image(self, image_name: str) -> Optional[str]:
        return self._q.board_images_get_board_for_image(image_name)

    def get_image_count_for_board(self, board_id: str) -> int:
        return self._q.board_images_get_image_count(board_id)

    def get_asset_count_for_board(self, board_id: str) -> int:
        return self._q.board_images_get_asset_count(board_id)
