"""Which boards an account may read: one rule for every query that lists boards or media through them."""

from sqlalchemy import ColumnElement, exists, or_

from invokeai.app.services.shared.database.schema.boards import board_members, boards


def readable_board(user_id: ColumnElement[str]) -> ColumnElement[bool]:
    """Boards an account that is not an administrator may read: its own, those every account gets a role on, and
    those it is a member of. Archived boards are not excluded here: listings decide that themselves."""
    member = exists().where(board_members.c.board_id == boards.c.board_id, board_members.c.user_id == user_id)
    return or_(boards.c.user_id == user_id, boards.c.general_access != "restricted", member)
