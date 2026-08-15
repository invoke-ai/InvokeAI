import sqlite3
from typing import Optional, Union, cast

from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordDeleteException,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordProjectOwnedException,
    BoardRecordSaveException,
    deserialize_board_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.misc import uuid_string

# Board ids per `IN (...)`. Well under SQLITE_MAX_VARIABLE_NUMBER on every build, including the
# 999-variable default of older SQLite.
_BOARD_ID_QUERY_CHUNK = 500


class SqliteBoardRecordStorage(BoardRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def delete_if_unclaimed(self, board_id: str) -> bool:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    DELETE FROM boards
                    WHERE board_id = ?
                      AND NOT EXISTS (
                        SELECT 1 FROM projects WHERE projects.board_id = boards.board_id
                      );
                    """,
                    (board_id,),
                )
                return cursor.rowcount > 0
            except Exception as e:
                raise BoardRecordDeleteException from e

    def get_project_ids_for_boards(self, board_ids: list[str]) -> dict[str, str]:
        if not board_ids:
            return {}

        # Chunked because the caller is a board *listing*: an admin's `GET /boards/?all=true`
        # passes every board on the install at once, and one bind parameter per board runs into
        # SQLITE_MAX_VARIABLE_NUMBER — which fails the whole listing rather than degrading.
        claimed: dict[str, str] = {}
        with self._db.transaction() as cursor:
            for start in range(0, len(board_ids), _BOARD_ID_QUERY_CHUNK):
                chunk = board_ids[start : start + _BOARD_ID_QUERY_CHUNK]
                placeholders = ", ".join("?" for _ in chunk)
                cursor.execute(
                    f"""--sql
                    SELECT board_id, project_id FROM projects WHERE board_id IN ({placeholders});
                    """,
                    tuple(chunk),
                )
                claimed.update({row[0]: row[1] for row in cursor.fetchall()})
        return claimed

    def get_with_project_id(self, board_id: str) -> tuple[BoardRecord, Optional[str]]:
        """The board and the project that claims it, in one query.

        `get_dto` needs both and is called on every authorization check, so asking for them
        separately doubled the round trips — and the transactions, which take a re-entrant lock —
        on the hottest read in the API.
        """
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT boards.*, projects.project_id AS claimed_by_project_id
                    FROM boards
                    LEFT JOIN projects ON projects.board_id = boards.board_id
                    WHERE boards.board_id = ?;
                    """,
                    (board_id,),
                )
                result = cast(Union[sqlite3.Row, None], cursor.fetchone())
            except sqlite3.Error as e:
                raise BoardRecordNotFoundException from e
        if result is None:
            raise BoardRecordNotFoundException
        row = dict(result)
        project_id = row.pop("claimed_by_project_id", None)
        return BoardRecord(**row), project_id

    def save(
        self,
        board_name: str,
        user_id: str,
    ) -> BoardRecord:
        with self._db.transaction() as cursor:
            try:
                board_id = uuid_string()
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO boards (board_id, board_name, user_id)
                    VALUES (?, ?, ?);
                    """,
                    (board_id, board_name, user_id),
                )
            except sqlite3.Error as e:
                raise BoardRecordSaveException from e
        return self.get(board_id)

    def get(
        self,
        board_id: str,
    ) -> BoardRecord:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT *
                    FROM boards
                    WHERE board_id = ?;
                    """,
                    (board_id,),
                )

                result = cast(Union[sqlite3.Row, None], cursor.fetchone())
            except sqlite3.Error as e:
                raise BoardRecordNotFoundException from e
        if result is None:
            raise BoardRecordNotFoundException
        return BoardRecord(**dict(result))

    def is_board_shared_with_user(self, board_id: str, user_id: str) -> bool:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT 1
                FROM shared_boards
                WHERE board_id = ? AND user_id = ?;
                """,
                (board_id, user_id),
            )

            return cursor.fetchone() is not None

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardRecord:
        with self._db.transaction() as cursor:
            try:
                changes_project_owned_state = (
                    changes.board_name is not None
                    or changes.archived is not None
                    or changes.board_visibility is not None
                )
                # One conditional write, not a read followed by a write. A project claim racing
                # this statement either commits first and makes rowcount zero, or waits until this
                # update has committed; there is no stale DTO window in which project-owned state
                # can be renamed, archived or published.
                cursor.execute(
                    """--sql
                    UPDATE boards
                    SET board_name = COALESCE(?, board_name),
                        cover_image_name = COALESCE(?, cover_image_name),
                        archived = COALESCE(?, archived),
                        board_visibility = COALESCE(?, board_visibility)
                    WHERE board_id = ?
                      AND (
                        ? = FALSE
                        OR NOT EXISTS (SELECT 1 FROM projects WHERE projects.board_id = boards.board_id)
                      );
                    """,
                    (
                        changes.board_name,
                        changes.cover_image_name,
                        changes.archived,
                        changes.board_visibility.value if changes.board_visibility is not None else None,
                        board_id,
                        changes_project_owned_state,
                    ),
                )

                if cursor.rowcount == 0:
                    cursor.execute("SELECT 1 FROM boards WHERE board_id = ?;", (board_id,))
                    if cursor.fetchone() is None:
                        raise BoardRecordNotFoundException
                    raise BoardRecordProjectOwnedException

            except sqlite3.Error as e:
                raise BoardRecordSaveException from e
        return self.get(board_id)

    def get_many(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardRecord]:
        with self._db.transaction() as cursor:
            # Build base query - admins see all boards, regular users see owned, shared, or public boards
            if is_admin:
                base_query = """
                    SELECT DISTINCT boards.*
                    FROM boards
                    {archived_filter}
                    ORDER BY {order_by} {direction}
                    LIMIT ? OFFSET ?;
                """

                # Determine archived filter condition
                archived_filter = "WHERE 1=1" if include_archived else "WHERE boards.archived = 0"

                final_query = base_query.format(
                    archived_filter=archived_filter, order_by=order_by.value, direction=direction.value
                )

                # Execute query to fetch boards
                cursor.execute(final_query, (limit, offset))
            else:
                base_query = """
                    SELECT DISTINCT boards.*
                    FROM boards
                    LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                    WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.board_visibility IN ('shared', 'public'))
                    {archived_filter}
                    ORDER BY {order_by} {direction}
                    LIMIT ? OFFSET ?;
                """

                # Determine archived filter condition
                archived_filter = "" if include_archived else "AND boards.archived = 0"

                final_query = base_query.format(
                    archived_filter=archived_filter, order_by=order_by.value, direction=direction.value
                )

                # Execute query to fetch boards
                cursor.execute(final_query, (user_id, user_id, limit, offset))

            result = cast(list[sqlite3.Row], cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]

            # Determine count query - admins count all boards, regular users count accessible boards
            if is_admin:
                if include_archived:
                    count_query = """
                        SELECT COUNT(DISTINCT boards.board_id)
                        FROM boards;
                    """
                else:
                    count_query = """
                        SELECT COUNT(DISTINCT boards.board_id)
                        FROM boards
                        WHERE boards.archived = 0;
                    """
                cursor.execute(count_query)
            else:
                if include_archived:
                    count_query = """
                        SELECT COUNT(DISTINCT boards.board_id)
                        FROM boards
                        LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.board_visibility IN ('shared', 'public'));
                    """
                else:
                    count_query = """
                        SELECT COUNT(DISTINCT boards.board_id)
                        FROM boards
                        LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.board_visibility IN ('shared', 'public'))
                        AND boards.archived = 0;
                    """

                # Execute count query
                cursor.execute(count_query, (user_id, user_id))

            count = cast(int, cursor.fetchone()[0])

        return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=count)

    def get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardRecord]:
        with self._db.transaction() as cursor:
            # Build query - admins see all boards, regular users see owned, shared, or public boards
            if is_admin:
                if order_by == BoardRecordOrderBy.Name:
                    base_query = """
                        SELECT DISTINCT boards.*
                        FROM boards
                        {archived_filter}
                        ORDER BY LOWER(boards.board_name) {direction}
                    """
                else:
                    base_query = """
                        SELECT DISTINCT boards.*
                        FROM boards
                        {archived_filter}
                        ORDER BY {order_by} {direction}
                    """

                archived_filter = "WHERE 1=1" if include_archived else "WHERE boards.archived = 0"

                final_query = base_query.format(
                    archived_filter=archived_filter, order_by=order_by.value, direction=direction.value
                )

                cursor.execute(final_query)
            else:
                if order_by == BoardRecordOrderBy.Name:
                    base_query = """
                        SELECT DISTINCT boards.*
                        FROM boards
                        LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.board_visibility IN ('shared', 'public'))
                        {archived_filter}
                        ORDER BY LOWER(boards.board_name) {direction}
                    """
                else:
                    base_query = """
                        SELECT DISTINCT boards.*
                        FROM boards
                        LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.board_visibility IN ('shared', 'public'))
                        {archived_filter}
                        ORDER BY {order_by} {direction}
                    """

                archived_filter = "" if include_archived else "AND boards.archived = 0"

                final_query = base_query.format(
                    archived_filter=archived_filter, order_by=order_by.value, direction=direction.value
                )

                cursor.execute(final_query, (user_id, user_id))

            result = cast(list[sqlite3.Row], cursor.fetchall())
        boards = [deserialize_board_record(dict(r)) for r in result]

        return boards
