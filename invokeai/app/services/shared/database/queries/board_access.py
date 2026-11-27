"""Which boards an account may read: one rule for every query that lists boards or media through them."""

from sqlalchemy import ColumnElement, bindparam, exists, false, or_

from invokeai.app.services.shared.database.schema.boards import board_members, boards


def readable_board(user_id: ColumnElement[str]) -> ColumnElement[bool]:
    """Boards an account that is not an administrator may read: its own, those every account gets a role on, and
    those it is a member of. Archived boards are not excluded here: listings decide that themselves."""
    member = exists().where(board_members.c.board_id == boards.c.board_id, board_members.c.user_id == user_id)
    return or_(boards.c.user_id == user_id, boards.c.general_access != "restricted", member)


def listed_board(scoped: bool) -> list[ColumnElement[bool]]:
    """The conditions on a board whose items a listing of every readable board shows: active, and with `scoped` one
    the account bound as `user_id` may read."""
    if not scoped:
        return [boards.c.archived == false()]
    return [boards.c.archived == false(), readable_board(bindparam("user_id"))]
