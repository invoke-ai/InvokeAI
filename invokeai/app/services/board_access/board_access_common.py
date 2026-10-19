"""Who may do what on a board: roles, the actions they allow, and the one table that maps them."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from invokeai.app.services.users.users_base import UserServiceBase


class BoardRole(str, Enum):
    """A caller's role on a board. Each role allows everything the roles before it allow."""

    Viewer = "viewer"
    Contributor = "contributor"
    Editor = "editor"
    Owner = "owner"


class GeneralAccess(str, Enum):
    """The role every other account gets on a board: none, viewer or contributor."""

    Restricted = "restricted"
    Viewer = "viewer"
    Contributor = "contributor"


class BoardAction(str, Enum):
    """What a caller can do to a board as a whole."""

    Read = "read"
    """Browse the board, open, download and copy out its items, and use them in one's own generations."""
    AddMedia = "add_media"
    """Upload to the board, generate into it, and move or copy one's own items onto it."""
    ManageItems = "manage_items"
    """Remove anyone's items from the board, and star or unstar them."""
    Rename = "rename"
    SetCover = "set_cover"
    Archive = "archive"
    SetGeneralAccess = "set_general_access"
    Move = "move"
    """Move the board between the Library and a project."""
    Delete = "delete"


class ItemAction(str, Enum):
    """What a caller can do to an image or a video. An item's owner may do all of it."""

    Read = "read"
    RemoveFromBoard = "remove_from_board"
    Star = "star"
    """Star or unstar: a flag every viewer of the item sees."""
    Edit = "edit"
    """Change the item's record otherwise (its category, whether it is an intermediate)."""
    Delete = "delete"
    """Delete the item permanently. Only its owner and admins may: a board's managers can only remove it."""


_RANK = {BoardRole.Viewer: 1, BoardRole.Contributor: 2, BoardRole.Editor: 3, BoardRole.Owner: 4}

MINIMUM_ROLE: dict[BoardAction, BoardRole] = {
    BoardAction.Read: BoardRole.Viewer,
    BoardAction.AddMedia: BoardRole.Contributor,
    BoardAction.ManageItems: BoardRole.Editor,
    BoardAction.Rename: BoardRole.Editor,
    BoardAction.SetCover: BoardRole.Editor,
    BoardAction.Archive: BoardRole.Editor,
    BoardAction.SetGeneralAccess: BoardRole.Owner,
    BoardAction.Move: BoardRole.Owner,
    BoardAction.Delete: BoardRole.Owner,
}
"""The least role each board action needs: the only copy of who may do what."""

# A project's inbox is managed through its project, by everyone, admins included.
_MANAGED_BY_PROJECT = frozenset(
    {BoardAction.Rename, BoardAction.Archive, BoardAction.SetGeneralAccess, BoardAction.Move, BoardAction.Delete}
)

# The board action that lets a caller do an item action to someone else's item. Deleting someone else's item
# permanently is for admins alone.
_BOARD_ACTION_FOR_ITEM: dict[ItemAction, Optional[BoardAction]] = {
    ItemAction.Read: BoardAction.Read,
    ItemAction.RemoveFromBoard: BoardAction.ManageItems,
    ItemAction.Star: BoardAction.ManageItems,
    ItemAction.Edit: None,
    ItemAction.Delete: None,
}


def role_of(
    *, owner_id: str, user_id: str, member_role: Optional[BoardRole], general_access: GeneralAccess
) -> Optional[BoardRole]:
    """The highest role a caller has on a board: as its owner, as a member, or through general access."""
    if owner_id == user_id:
        return BoardRole.Owner
    candidates = [member_role] if member_role is not None else []
    if general_access is not GeneralAccess.Restricted:
        candidates.append(BoardRole(general_access.value))
    return max(candidates, key=_RANK.__getitem__, default=None)


@dataclass(frozen=True)
class BoardAccess:
    """A caller's access to one board."""

    role: Optional[BoardRole]
    """The caller's own role, or None when only admin rights give access."""
    is_admin: bool
    is_inbox: bool

    @property
    def admin_override(self) -> bool:
        """Whether admin rights give the caller more than their own role does."""
        return self.is_admin and self.role is not BoardRole.Owner

    def allows(self, action: BoardAction) -> bool:
        return not (self.is_inbox and action in _MANAGED_BY_PROJECT) and self.role_allows(action)

    def role_allows(self, action: BoardAction) -> bool:
        """Whether the caller's role, or admin rights, allow the action, setting aside what a project's inbox
        reserves to its project: for a caller that leaves that refusal to the storage."""
        if self.is_admin:
            return True
        return self.role is not None and _RANK[self.role] >= _RANK[MINIMUM_ROLE[action]]

    def allowed_actions(self) -> list[BoardAction]:
        return [action for action in BoardAction if self.allows(action)]


def item_allows(*, is_item_owner: bool, is_admin: bool, board: Optional[BoardAccess], action: ItemAction) -> bool:
    """Whether a caller may do an item action, given the access they have to the board the item is on (None when
    it is on none)."""
    if is_item_owner or is_admin:
        return True
    board_action = _BOARD_ACTION_FOR_ITEM[action]
    return board is not None and board_action is not None and board.allows(board_action)


@dataclass(frozen=True)
class BoardViewer:
    """The account a board is described to."""

    user_id: str
    is_admin: bool


def queue_viewer(user_id: str, *, multiuser: bool, users: "UserServiceBase") -> Optional[BoardViewer]:
    """The queue user as the boards API sees a caller, or None when the account may do nothing at queue time.

    In single-user mode the queue runs as an admin and the account is not looked up. In multiuser mode a deleted or
    deactivated account keeps no queue-time privileges, however its item got past the processor's owner checks, and
    an active one brings its own admin flag. Graph execution and workflow-call expansion both resolve the queue
    user through here, so a refusal at queue time is the same refusal everywhere.
    """
    if not multiuser:
        return BoardViewer(user_id=user_id, is_admin=True)
    user = users.get(user_id)
    if user is None or not user.is_active:
        return None
    return BoardViewer(user_id=user_id, is_admin=user.is_admin)


@dataclass(frozen=True)
class BoardAudience:
    """Who sees what lands on a board: everyone, or its owner and members (admins see every board)."""

    everyone: bool
    user_ids: tuple[str, ...]


class BoardAccessDeniedError(Exception):
    """The caller may not do the action to the board."""

    def __init__(self, board_id: str, action: BoardAction) -> None:
        super().__init__(f"Not allowed to {action.value} on board {board_id}")
        self.board_id = board_id
        self.action = action


class ItemAccessDeniedError(Exception):
    """The caller may not do the action to the item."""

    def __init__(self, item_name: str, action: ItemAction) -> None:
        super().__init__(f"Not allowed to {action.value} {item_name}")
        self.item_name = item_name
        self.action = action
