"""Boards and who they are shared with."""

import functools
import itertools
from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    Update,
    bindparam,
    case,
    delete,
    exists,
    false,
    func,
    insert,
    literal,
    select,
    update,
)

from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordOrderBy,
    BoardVisibility,
)
from invokeai.app.services.shared.database.dialect import CaseInsensitiveOrder
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, locking, mapped, read, write
from invokeai.app.services.shared.database.queries.board_access import readable_board
from invokeai.app.services.shared.database.schema.boards import boards, shared_boards
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.pagination import SQLiteDirection

_BOARD_COLUMNS = (
    boards.c.board_id,
    boards.c.board_name,
    boards.c.user_id,
    boards.c.created_at,
    boards.c.updated_at,
    boards.c.deleted_at,
    boards.c.cover_image_name,
    boards.c.archived,
    boards.c.board_visibility,
    boards.c.project_id,
)
_CLAIMED = exists().where(projects.c.board_id == boards.c.board_id)

_GET = select(*_BOARD_COLUMNS).where(boards.c.board_id == bindparam("board_id"))
_GET_WITH_INBOX_PROJECT = (
    select(*_BOARD_COLUMNS, projects.c.project_id)
    .select_from(boards.outerjoin(projects, projects.c.board_id == boards.c.board_id))
    .where(boards.c.board_id == bindparam("board_id"))
)
_EXISTS = select(literal(1)).where(boards.c.board_id == bindparam("board_id"))
_LOCK = (
    select(boards.c.user_id, boards.c.board_visibility)
    .where(boards.c.board_id == bindparam("board_id"))
    .with_for_update()
)
_PROJECT_IDS = select(projects.c.board_id, projects.c.project_id).where(
    projects.c.board_id.in_(bindparam("board_ids", expanding=True))
)
_INSERT = insert(boards)
_DELETE_IF_UNCLAIMED = delete(boards).where(boards.c.board_id == bindparam("board_id"), ~_CLAIMED)
_IS_SHARED_WITH = select(literal(1)).where(
    shared_boards.c.board_id == bindparam("board_id"), shared_boards.c.user_id == bindparam("user_id")
)
_SHARED_USER_IDS = select(shared_boards.c.user_id).where(shared_boards.c.board_id == bindparam("board_id"))
_IS_SHARED = select(exists().where(shared_boards.c.board_id == bindparam("board_id")))
_RENAME = (
    update(boards)
    .where(boards.c.board_id == bindparam("target_board_id"))
    .values(board_name=bindparam("new_board_name"))
)
_RENAME_AND_UNARCHIVE = _RENAME.values(archived=false())
_SET_PROJECT = (
    update(boards)
    .where(boards.c.board_id == bindparam("target_board_id"))
    .values(project_id=bindparam("new_project_id"))
)
_OTHER_MEMBERS = (
    boards.c.user_id == bindparam("target_user_id"),
    boards.c.project_id == bindparam("target_project_id"),
    boards.c.board_id != bindparam("inbox_id"),
)
_RELEASE_MEMBERS = update(boards).where(*_OTHER_MEMBERS).values(project_id=None)
_DELETE_MEMBERS = delete(boards).where(*_OTHER_MEMBERS)


def _board(row: Sequence[Any]) -> BoardRecord:
    (
        board_id,
        board_name,
        user_id,
        created_at,
        updated_at,
        deleted_at,
        cover_image_name,
        archived,
        visibility,
        project_id,
    ) = row
    try:
        board_visibility = BoardVisibility(visibility)
    except ValueError:
        board_visibility = BoardVisibility.Private
    return BoardRecord(
        board_id=board_id,
        board_name=board_name,
        user_id=user_id,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        cover_image_name=cover_image_name,
        archived=archived,
        board_visibility=board_visibility,
        project_id=project_id,
    )


def _board_or_none(row: Optional[Sequence[Any]]) -> Optional[BoardRecord]:
    return _board(row) if row is not None else None


def _board_and_project_id(row: Optional[Sequence[Any]]) -> Optional[tuple[BoardRecord, Optional[str]]]:
    return (_board(row[:-1]), row[-1]) if row is not None else None


def _boards(rows: Sequence[Sequence[Any]]) -> list[BoardRecord]:
    return [_board(row) for row in rows]


def _boards_and_total(result: tuple[Sequence[Sequence[Any]], int]) -> tuple[list[BoardRecord], int]:
    rows, total = result
    return _boards(rows), total


@functools.cache
def _update(changes_owned_state: bool) -> Update:
    """Sets the given fields (a NULL leaves one as it is). A change to what a project owns -- the name, the
    archived flag, the visibility -- applies only to a board no project claims."""
    statement = (
        update(boards)
        .where(boards.c.board_id == bindparam("target_board_id"))
        .values(
            board_name=func.coalesce(bindparam("new_board_name", type_=boards.c.board_name.type), boards.c.board_name),
            cover_image_name=func.coalesce(
                bindparam("new_cover_image_name", type_=boards.c.cover_image_name.type), boards.c.cover_image_name
            ),
            archived=func.coalesce(bindparam("new_archived", type_=boards.c.archived.type), boards.c.archived),
            project_id=case(
                (bindparam("moves_board"), bindparam("new_project_id", type_=boards.c.project_id.type)),
                else_=boards.c.project_id,
            ),
            board_visibility=func.coalesce(
                bindparam("new_board_visibility", type_=boards.c.board_visibility.type), boards.c.board_visibility
            ),
        )
    )
    return statement.where(~_CLAIMED) if changes_owned_state else statement


def _listing(statement: Select[Any], is_admin: bool, include_archived: bool) -> Select[Any]:
    if not is_admin:
        statement = statement.where(readable_board(bindparam("user_id")))
    if not include_archived:
        statement = statement.where(boards.c.archived == false())
    return statement


@functools.cache
def _list(
    is_admin: bool,
    include_archived: bool,
    order_by: BoardRecordOrderBy,
    direction: SQLiteDirection,
    paged: bool,
) -> Select[Any]:
    # The enums are also strings, which share their cache entries: compare by value, never by identity.
    # `get_all` has always sorted names case-insensitively, a page of `get_many` by the names as stored.
    key: ColumnElement[Any] = boards.c.board_name if order_by == BoardRecordOrderBy.Name else boards.c.created_at
    if order_by == BoardRecordOrderBy.Name and not paged:
        key = CaseInsensitiveOrder(boards.c.board_name)
    # The board id breaks ties, so that equal keys keep one order from page to page.
    ordering = [key, boards.c.board_id]
    statement = _listing(select(*_BOARD_COLUMNS), is_admin, include_archived).order_by(
        *(column.desc() if direction == SQLiteDirection.Descending else column.asc() for column in ordering)
    )
    if paged:
        statement = statement.limit(bindparam("limit")).offset(bindparam("offset"))
    return statement


@functools.cache
def _count(is_admin: bool, include_archived: bool) -> Select[Any]:
    return _listing(select(func.count()).select_from(boards), is_admin, include_archived)


class BoardQueries(QueryModule):
    @mapped(_board_or_none)
    @read
    def get(self, conn: Connection, board_id: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"board_id": board_id}).first()

    @mapped(_board_and_project_id)
    @read
    def get_with_inbox_project(self, conn: Connection, board_id: str) -> Optional[Row[Any]]:
        """The board and the id of the project that claims it, if one does."""
        return conn.execute(_GET_WITH_INBOX_PROJECT, {"board_id": board_id}).first()

    @read
    def exists(self, conn: Connection, board_id: str) -> bool:
        return conn.execute(_EXISTS, {"board_id": board_id}).first() is not None

    @locking
    def lock(self, conn: Connection, board_id: str) -> Optional[tuple[str, str]]:
        """Locks the board's row until the transaction ends: its owner and visibility, or None when there is no
        such board.

        Locked before a transaction reads what it decides on, the row makes every transaction that changes the
        board -- its owner, visibility or sharing, a project claiming it, its deletion -- wait until this one has
        ended, and this one's later reads see what such a transaction committed before. Lock the board's project
        first, if there is one: rows are locked in that order.
        """
        row: Optional[Sequence[Any]] = conn.execute(_LOCK, {"board_id": board_id}).first()
        if row is None:
            return None
        user_id, board_visibility = row
        return user_id, board_visibility

    @read
    def project_ids(self, conn: Connection, board_ids: Sequence[str]) -> dict[str, str]:
        """The id of the project that claims each of these boards, for the boards a project claims."""
        claimed: dict[str, str] = {}
        for chunk in itertools.batched(board_ids, IN_CHUNK):
            claimed.update((row[0], row[1]) for row in conn.execute(_PROJECT_IDS, {"board_ids": list(chunk)}).all())
        return claimed

    @mapped(_boards_and_total)
    @read
    def page(
        self,
        conn: Connection,
        *,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int,
        limit: int,
        include_archived: bool,
    ) -> tuple[Sequence[Row[Any]], int]:
        """A page of the boards the user sees, and how many there are in all."""
        parameters = {"user_id": user_id, "limit": limit, "offset": offset}
        rows = conn.execute(_list(is_admin, include_archived, order_by, direction, True), parameters).all()
        total: int = conn.execute(_count(is_admin, include_archived), parameters).scalar_one()
        return rows, total

    @mapped(_boards)
    @read
    def all(
        self,
        conn: Connection,
        *,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool,
    ) -> Sequence[Row[Any]]:
        """Every board the user sees."""
        statement = _list(is_admin, include_archived, order_by, direction, False)
        return conn.execute(statement, {"user_id": user_id}).all()

    @read
    def is_shared_with(self, conn: Connection, board_id: str, user_id: str) -> bool:
        return conn.execute(_IS_SHARED_WITH, {"board_id": board_id, "user_id": user_id}).first() is not None

    @read
    def shared_user_ids(self, conn: Connection, board_id: str) -> list[str]:
        return list(conn.execute(_SHARED_USER_IDS, {"board_id": board_id}).scalars().all())

    @read
    def is_shared(self, conn: Connection, board_id: str) -> bool:
        """Whether the board is shared with any account."""
        return bool(conn.execute(_IS_SHARED, {"board_id": board_id}).scalar_one())

    @write
    def insert(
        self, conn: Connection, *, board_id: str, board_name: str, user_id: str, project_id: str | None = None
    ) -> None:
        conn.execute(
            _INSERT, {"board_id": board_id, "board_name": board_name, "user_id": user_id, "project_id": project_id}
        )

    @write
    def update(self, conn: Connection, board_id: str, changes: BoardChanges) -> bool:
        """Applies the changes; whether a board took them. A board a project claims keeps its name, archived flag
        and visibility, so a change to one of those leaves it as it is."""
        changes_owned_state = (
            changes.board_name is not None
            or changes.archived is not None
            or changes.board_visibility is not None
            or changes.moves_board
        )
        result = conn.execute(
            _update(changes_owned_state),
            {
                "target_board_id": board_id,
                "moves_board": changes.moves_board,
                "new_project_id": changes.project_id,
                "new_board_name": changes.board_name,
                "new_cover_image_name": changes.cover_image_name,
                "new_archived": changes.archived,
                "new_board_visibility": (
                    changes.board_visibility.value if changes.board_visibility is not None else None
                ),
            },
        )
        return result.rowcount > 0

    @write
    def rename(self, conn: Connection, board_id: str, board_name: str, *, unarchive: bool = False) -> None:
        """Renames the board, and with `unarchive` un-archives it, whether or not a project claims it: for the
        projects storage, whose boards take their names from their projects. (`update` leaves the name of a
        claimed board as it is.)"""
        statement = _RENAME_AND_UNARCHIVE if unarchive else _RENAME
        conn.execute(statement, {"target_board_id": board_id, "new_board_name": board_name})

    @write
    def set_project(self, conn: Connection, board_id: str, project_id: str) -> None:
        """Sets membership for an inbox claim; the caller holds the board's lock."""
        conn.execute(_SET_PROJECT, {"target_board_id": board_id, "new_project_id": project_id})

    @write
    def remove_project_members(
        self, conn: Connection, user_id: str, project_id: str, inbox_id: str, *, delete_members: bool
    ) -> None:
        """Releases or deletes the owner's non-inbox members while the project is locked. Media remain intact."""
        conn.execute(
            _DELETE_MEMBERS if delete_members else _RELEASE_MEMBERS,
            {"target_user_id": user_id, "target_project_id": project_id, "inbox_id": inbox_id},
        )

    @write
    def delete_if_unclaimed(self, conn: Connection, board_id: str) -> bool:
        """Deletes the board unless a project claims it; whether it did."""
        return conn.execute(_DELETE_IF_UNCLAIMED, {"board_id": board_id}).rowcount > 0
