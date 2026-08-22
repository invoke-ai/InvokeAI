from typing import Optional

from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecord, BoardRecordOrderBy
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

        # Match the gallery's deterministic (starred, created_at, kind, name) ordering.
        image_key = (cover_image.starred, cover_image.created_at, "image", cover_image.image_name)
        video_key = (cover_video.starred, cover_video.created_at, "video", cover_video.video_name)
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
        # One query for the record and its claiming project. `get_dto` is what every authorization
        # check resolves through, so a second lookup here is a second transaction — and the lock it
        # takes — on the most-travelled read in the API.
        board_record, project_id = self.__invoker.services.board_records.get_with_project_id(board_id)
        cover_image_name, cover_video_name = self._resolve_cover(board_record.board_id)
        image_count, video_count, asset_count = self._get_counts(board_id)
        return board_record_to_dto(
            board_record,
            cover_image_name,
            image_count,
            asset_count,
            cover_video_name=cover_video_name,
            video_count=video_count,
            project_id=project_id,
        )

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardDTO:
        self.__invoker.services.board_records.update(board_id, changes)
        # Re-read through `get_dto` rather than shaping the update's own return: it is the one
        # place that resolves cover, counts and claiming project together, and an update that
        # dropped `project_id` would tell the gallery a project's board is an ordinary one.
        return self.get_dto(board_id)

    def delete_if_unclaimed(self, board_id: str) -> bool:
        return self.__invoker.services.board_records.delete_if_unclaimed(board_id)

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
        board_dtos = self._to_dtos(board_records.items, is_admin)

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
        return self._to_dtos(board_records, is_admin)

    def _to_dtos(self, board_records: list[BoardRecord], is_admin: bool) -> list[BoardDTO]:
        """Builds board DTOs for a listing with a fixed number of queries.

        Both the media summaries and (for admins) the owner display names are fetched for
        the whole page at once. The owner lookup used to run one `users.get` per board, so
        an admin listing 50 boards issued 50 extra queries for what is usually a handful of
        distinct owners.
        """
        summaries = self.__invoker.services.gallery.get_board_media_summaries(
            [record.board_id for record in board_records]
        )
        project_ids = self.__invoker.services.board_records.get_project_ids_for_boards(
            [record.board_id for record in board_records]
        )
        owners = (
            self.__invoker.services.users.get_many([record.user_id for record in board_records]) if is_admin else {}
        )

        board_dtos: list[BoardDTO] = []
        for r in board_records:
            summary = summaries[r.board_id]

            # For admin users, include owner username
            owner = owners.get(r.user_id)
            owner_username = (owner.display_name or owner.email) if owner else None

            board_dtos.append(
                board_record_to_dto(
                    r,
                    summary.cover_image_name,
                    summary.image_count,
                    summary.asset_count,
                    owner_username,
                    cover_video_name=summary.cover_video_name,
                    video_count=summary.video_count,
                    project_id=project_ids.get(r.board_id),
                )
            )

        return board_dtos
