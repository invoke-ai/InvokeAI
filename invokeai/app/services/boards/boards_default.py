from typing import Optional

from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecordOrderBy
from invokeai.app.services.boards.boards_base import BoardServiceABC
from invokeai.app.services.boards.boards_common import BoardDTO, board_record_to_dto
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class BoardService(BoardServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def _resolve_cover(self, board_id: str) -> tuple[Optional[str], Optional[str]]:
        """Pick the cover item for a board, considering both images and videos.

        Returns ``(cover_image_name, cover_video_name)`` — at most one is set.
        The winner is chosen by ``(starred DESC, created_at DESC)`` across both
        tables so a recent video can supersede an older image (and vice versa).
        """
        cover_image = self.__invoker.services.image_records.get_most_recent_image_for_board(board_id)
        cover_video = self.__invoker.services.video_records.get_most_recent_video_for_board(board_id)

        if cover_image is None and cover_video is None:
            return None, None
        if cover_video is None:
            assert cover_image is not None
            return cover_image.image_name, None
        if cover_image is None:
            return None, cover_video.video_name

        # Both candidates exist — compare on (starred, created_at).
        image_key = (cover_image.starred, cover_image.created_at)
        video_key = (cover_video.starred, cover_video.created_at)
        if video_key > image_key:
            return None, cover_video.video_name
        return cover_image.image_name, None

    def _get_counts(self, board_id: str) -> tuple[int, int, int]:
        """Return ``(image_count, video_count, asset_count)`` for a board."""
        image_count = self.__invoker.services.board_image_records.get_image_count_for_board(board_id)
        asset_count = self.__invoker.services.board_image_records.get_asset_count_for_board(board_id)
        video_count = self.__invoker.services.board_video_records.get_video_count_for_board(board_id)
        return image_count, video_count, asset_count

    def create(
        self,
        board_name: str,
        user_id: str,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.save(board_name, user_id)
        return board_record_to_dto(board_record, None, 0, 0)

    def get_dto(self, board_id: str) -> BoardDTO:
        board_record = self.__invoker.services.board_records.get(board_id)
        cover_image_name, cover_video_name = self._resolve_cover(board_record.board_id)
        image_count, video_count, asset_count = self._get_counts(board_id)
        return board_record_to_dto(
            board_record,
            cover_image_name,
            image_count,
            asset_count,
            cover_video_name=cover_video_name,
            video_count=video_count,
        )

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.update(board_id, changes)
        cover_image_name, cover_video_name = self._resolve_cover(board_record.board_id)
        image_count, video_count, asset_count = self._get_counts(board_id)
        return board_record_to_dto(
            board_record,
            cover_image_name,
            image_count,
            asset_count,
            cover_video_name=cover_video_name,
            video_count=video_count,
        )

    def delete(self, board_id: str) -> None:
        self.__invoker.services.board_records.delete(board_id)

    def get_many(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardDTO]:
        board_records = self.__invoker.services.board_records.get_many(
            user_id, is_admin, order_by, direction, offset, limit, include_archived
        )
        board_dtos = []
        for r in board_records.items:
            cover_image_name, cover_video_name = self._resolve_cover(r.board_id)
            image_count, video_count, asset_count = self._get_counts(r.board_id)

            # For admin users, include owner username
            owner_username = None
            if is_admin:
                owner = self.__invoker.services.users.get(r.user_id)
                if owner:
                    owner_username = owner.display_name or owner.email

            board_dtos.append(
                board_record_to_dto(
                    r,
                    cover_image_name,
                    image_count,
                    asset_count,
                    owner_username,
                    cover_video_name=cover_video_name,
                    video_count=video_count,
                )
            )

        return OffsetPaginatedResults[BoardDTO](items=board_dtos, offset=offset, limit=limit, total=len(board_dtos))

    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardDTO]:
        board_records = self.__invoker.services.board_records.get_all(
            user_id, is_admin, order_by, direction, include_archived
        )
        board_dtos = []
        for r in board_records:
            cover_image_name, cover_video_name = self._resolve_cover(r.board_id)
            image_count, video_count, asset_count = self._get_counts(r.board_id)

            # For admin users, include owner username
            owner_username = None
            if is_admin:
                owner = self.__invoker.services.users.get(r.user_id)
                if owner:
                    owner_username = owner.display_name or owner.email

            board_dtos.append(
                board_record_to_dto(
                    r,
                    cover_image_name,
                    image_count,
                    asset_count,
                    owner_username,
                    cover_video_name=cover_video_name,
                    video_count=video_count,
                )
            )

        return board_dtos
