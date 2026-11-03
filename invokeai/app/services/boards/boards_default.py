from typing import Optional

from invokeai.app.services.board_access.board_access_common import BoardAccess, BoardRole, BoardViewer, role_of
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardRecord, BoardRecordOrderBy
from invokeai.app.services.boards.boards_base import BoardServiceABC
from invokeai.app.services.boards.boards_common import BoardDTO, board_record_to_dto
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection


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

    def _get_counts(self, board_id: str) -> tuple[int, int, int, int]:
        """Return ``(image_count, video_count, asset_count, asset_video_count)`` for a board."""
        image_count, asset_count = self.__invoker.services.board_image_records.get_counts_for_board(board_id)
        video_count, asset_video_count = self.__invoker.services.board_video_records.get_counts_for_board(board_id)
        return image_count, video_count, asset_count, asset_video_count

    def create(
        self,
        board_name: str,
        user_id: str,
        project_id: Optional[str] = None,
        *,
        viewer: BoardViewer,
    ) -> BoardDTO:
        board_record = self.__invoker.services.board_records.save(board_name, user_id, project_id)
        return self._describe([board_record_to_dto(board_record, None, 0, 0)], viewer)[0]

    def get_dto(self, board_id: str, viewer: BoardViewer) -> BoardDTO:
        # One query for the record and the project it is the inbox of: a second lookup would be a second
        # transaction, and the lock it takes, on a read every board route makes.
        board_record, inbox_of_project_id = self.__invoker.services.board_records.get_with_inbox_project(board_id)
        cover_image_name, cover_video_name = self._resolve_cover(board_record.board_id)
        image_count, video_count, asset_count, asset_video_count = self._get_counts(board_id)
        dto = board_record_to_dto(
            board_record,
            cover_image_name,
            image_count,
            asset_count,
            cover_video_name=cover_video_name,
            video_count=video_count,
            asset_video_count=asset_video_count,
            is_inbox=inbox_of_project_id is not None,
        )
        return self._describe([dto], viewer)[0]

    def update(self, board_id: str, changes: BoardChanges, viewer: BoardViewer) -> BoardDTO:
        self.__invoker.services.board_records.update(board_id, changes)
        # Re-read through `get_dto` rather than shaping the update's own return: it is the one
        # place that resolves cover, counts and the inbox flag together, and an update that
        # dropped the flag would tell the gallery a project's inbox is an ordinary board.
        return self.get_dto(board_id, viewer)

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
        board_dtos = self._to_dtos(board_records.items, BoardViewer(user_id=user_id, is_admin=is_admin))

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
        return self._to_dtos(board_records, BoardViewer(user_id=user_id, is_admin=is_admin))

    def _to_dtos(self, board_records: list[BoardRecord], viewer: BoardViewer) -> list[BoardDTO]:
        """Builds the board DTOs of a listing with a fixed number of queries: the media summaries and inbox flags
        for the whole page at once, then what `_describe` adds."""
        summaries = self.__invoker.services.gallery.get_board_media_summaries(
            [record.board_id for record in board_records]
        )
        inbox_board_ids = self.__invoker.services.board_records.get_inbox_board_ids(
            [record.board_id for record in board_records]
        )
        dtos = [
            board_record_to_dto(
                r,
                summaries[r.board_id].cover_image_name,
                summaries[r.board_id].image_count,
                summaries[r.board_id].asset_count,
                cover_video_name=summaries[r.board_id].cover_video_name,
                video_count=summaries[r.board_id].video_count,
                asset_video_count=summaries[r.board_id].asset_video_count,
                is_inbox=r.board_id in inbox_board_ids,
            )
            for r in board_records
        ]
        return self._describe(dtos, viewer)

    def _describe(self, dtos: list[BoardDTO], viewer: BoardViewer) -> list[BoardDTO]:
        """Adds what depends on who is looking: their role and allowed actions, the owner's name, and the project,
        which only the owner and admins learn. Two queries for any number of boards."""
        board_ids = [dto.board_id for dto in dtos]
        member_roles = self.__invoker.services.board_access.member_roles(board_ids, viewer.user_id)
        owners = self.__invoker.services.users.get_many([dto.user_id for dto in dtos])
        described: list[BoardDTO] = []
        for dto in dtos:
            role = role_of(
                owner_id=dto.user_id,
                user_id=viewer.user_id,
                member_role=member_roles.get(dto.board_id),
                general_access=dto.general_access,
            )
            access = BoardAccess(role=role, is_admin=viewer.is_admin, is_inbox=dto.is_inbox)
            owner = owners.get(dto.user_id)
            owner_username = None
            if owner is not None:
                # An email only for admins and the owner themselves.
                sees_email = viewer.is_admin or owner.user_id == viewer.user_id
                owner_username = owner.display_name or (owner.email if sees_email else "Unknown user")
            sees_project = viewer.is_admin or role is BoardRole.Owner
            described.append(
                dto.model_copy(
                    update={
                        "my_role": role,
                        "allowed_actions": access.allowed_actions(),
                        "admin_override": access.admin_override,
                        "owner_username": owner_username,
                        "project_id": dto.project_id if sees_project else None,
                    }
                )
            )
        return described
