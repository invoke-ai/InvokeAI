from typing import Optional

from invokeai.app.services.board_video_records.board_video_records_base import BoardVideoRecordStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.database.database import Database


class BoardVideoRecordStorage(BoardVideoRecordStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def add_video_to_board(self, board_id: str, video_name: str) -> None:
        self._queries.board_videos.add(board_id, video_name)

    def remove_video_from_board(self, video_name: str) -> None:
        self._queries.board_videos.remove(video_name)

    def get_all_board_video_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
        user_id: Optional[str] = None,
    ) -> list[str]:
        # `board_id` "none" means the videos on no board; admins pass user_id=None to see every account's.
        return self._queries.board_videos.video_names(
            None if board_id == "none" else board_id,
            categories=categories,
            is_intermediate=is_intermediate,
            user_id=user_id,
        )

    def get_board_for_video(self, video_name: str) -> Optional[str]:
        return self._queries.board_videos.board_of(video_name)

    def get_counts_for_board(self, board_id: str) -> tuple[int, int]:
        return self._queries.board_videos.counts(board_id)
