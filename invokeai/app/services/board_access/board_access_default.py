from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from invokeai.app.services.board_access.board_access_common import (
    BoardAccess,
    BoardAccessDeniedError,
    BoardAction,
    BoardAudience,
    BoardRole,
    GeneralAccess,
    ItemAccessDeniedError,
    ItemAction,
    item_allows,
    role_of,
)
from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries

if TYPE_CHECKING:
    from invokeai.app.services.invoker import Invoker


class BoardAccessService:
    """Decides what a caller may do to a board and to the images and videos on it.

    Routers, graph execution, workflow-call expansion and socket routing all ask here, so the rules in
    `board_access_common` are applied the same way everywhere. Listings decide which boards a caller sees in SQL
    (`queries/board_access.readable_board`), by the same rules.
    """

    __invoker: Invoker

    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def _queries(self) -> Queries:
        # Looked up on use: `InvocationServices` builds this service for every caller, some of them with no
        # database to query.
        return self._database.queries

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def get(self, board_id: str, user_id: str, is_admin: bool) -> BoardAccess:
        """The caller's access to the board. Raises `BoardRecordNotFoundException` when there is no such board."""
        facts = self._queries.boards.access_facts(board_id, user_id)
        if facts is None:
            raise BoardRecordNotFoundException
        owner_id, general_access, member_role, is_inbox = facts
        role = role_of(
            owner_id=owner_id,
            user_id=user_id,
            member_role=BoardRole(member_role) if member_role is not None else None,
            general_access=GeneralAccess(general_access),
        )
        return BoardAccess(role=role, is_admin=is_admin, is_inbox=is_inbox)

    def require(self, board_id: str, user_id: str, is_admin: bool, action: BoardAction) -> BoardAccess:
        """The caller's access, when it allows the action. Raises `BoardAccessDeniedError` otherwise, and
        `BoardRecordNotFoundException` when there is no such board."""
        access = self.get(board_id, user_id, is_admin)
        if not access.allows(action):
            raise BoardAccessDeniedError(board_id, action)
        return access

    def member_roles(self, board_ids: list[str], user_id: str) -> dict[str, BoardRole]:
        """The caller's membership role on each of these boards it is a member of, in one query."""
        return {
            board_id: BoardRole(role)
            for board_id, role in self._queries.boards.member_roles(board_ids, user_id).items()
        }

    def require_image(self, image_name: str, user_id: str, is_admin: bool, action: ItemAction) -> None:
        """Raises `ItemAccessDeniedError` unless the caller may do the action to the image."""
        if is_admin:
            return
        services = self.__invoker.services
        is_owner = services.image_records.get_user_id(image_name) == user_id
        board_id = None if is_owner else services.board_image_records.get_board_for_image(image_name)
        self._require_item(image_name, board_id, user_id, is_admin, is_owner, action)

    def require_video(self, video_name: str, user_id: str, is_admin: bool, action: ItemAction) -> None:
        """Raises `ItemAccessDeniedError` unless the caller may do the action to the video."""
        if is_admin:
            return
        services = self.__invoker.services
        is_owner = services.video_records.get_user_id(video_name) == user_id
        board_id = None if is_owner else services.board_video_records.get_board_for_video(video_name)
        self._require_item(video_name, board_id, user_id, is_admin, is_owner, action)

    def _require_item(
        self,
        item_name: str,
        board_id: Optional[str],
        user_id: str,
        is_admin: bool,
        is_owner: bool,
        action: ItemAction,
    ) -> None:
        board: Optional[BoardAccess] = None
        if board_id is not None:
            try:
                board = self.get(board_id, user_id, is_admin)
            except BoardRecordNotFoundException:
                board = None
        if not item_allows(is_item_owner=is_owner, is_admin=is_admin, board=board, action=action):
            raise ItemAccessDeniedError(item_name, action)

    def visible_project_ids(self, project_ids: Iterable[str], user_id: str, is_admin: bool) -> set[str]:
        """The project ids among these the caller may learn from the media made in them: all of them for an admin,
        otherwise those naming the caller's own projects. One query, however many ids."""
        ids = set(project_ids)
        if is_admin or not ids:
            return ids
        return self._queries.projects.owned_ids(user_id, sorted(ids))

    def audience(self, board_id: str) -> Optional[BoardAudience]:
        """Who sees what lands on the board, or None when there is no such board."""
        record = self._queries.boards.get(board_id)
        if record is None:
            return None
        if GeneralAccess(record.general_access) is not GeneralAccess.Restricted:
            return BoardAudience(everyone=True, user_ids=())
        members = self._queries.boards.shared_user_ids(board_id)
        return BoardAudience(everyone=False, user_ids=(record.user_id, *members))
