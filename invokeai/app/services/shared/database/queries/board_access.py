"""Which boards an account may read: one rule for every query that lists boards or media through them."""

from sqlalchemy import ColumnElement, exists, literal, or_

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.shared.database.schema.boards import boards, shared_boards

_SHARED_VISIBILITIES = (literal(BoardVisibility.Shared.value), literal(BoardVisibility.Public.value))


def readable_board(user_id: ColumnElement[str]) -> ColumnElement[bool]:
    """Boards an account that is not an administrator may read: its own, those shared or public, and those shared
    with it. Archived boards are not excluded here: listings decide that themselves."""
    shared_with_account = exists().where(
        shared_boards.c.board_id == boards.c.board_id, shared_boards.c.user_id == user_id
    )
    return or_(boards.c.user_id == user_id, boards.c.board_visibility.in_(_SHARED_VISIBILITIES), shared_with_account)
