import sqlite3
from typing import Union, cast

from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordDeleteException,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordSaveException,
    deserialize_board_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.misc import uuid_string


class SqliteBoardRecordStorage(BoardRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def delete(self, board_id: str) -> None:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    DELETE FROM boards
                    WHERE board_id = ?;
                    """,
                    (board_id,),
                )
            except Exception as e:
                raise BoardRecordDeleteException from e

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

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardRecord:
        with self._db.transaction() as cursor:
            try:
                # Change the name of a board
                if changes.board_name is not None:
                    cursor.execute(
                        """--sql
                        UPDATE boards
                        SET board_name = ?
                        WHERE board_id = ?;
                        """,
                        (changes.board_name, board_id),
                    )

                # Change the cover image of a board
                if changes.cover_image_name is not None:
                    cursor.execute(
                        """--sql
                        UPDATE boards
                        SET cover_image_name = ?
                        WHERE board_id = ?;
                        """,
                        (changes.cover_image_name, board_id),
                    )

                # Change the archived status of a board
                if changes.archived is not None:
                    cursor.execute(
                        """--sql
                        UPDATE boards
                        SET archived = ?
                        WHERE board_id = ?;
                        """,
                        (changes.archived, board_id),
                    )

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
                    WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.is_public = 1)
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
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.is_public = 1);
                    """
                else:
                    count_query = """
                        SELECT COUNT(DISTINCT boards.board_id)
                        FROM boards
                        LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.is_public = 1)
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
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.is_public = 1)
                        {archived_filter}
                        ORDER BY LOWER(boards.board_name) {direction}
                    """
                else:
                    base_query = """
                        SELECT DISTINCT boards.*
                        FROM boards
                        LEFT JOIN shared_boards ON boards.board_id = shared_boards.board_id
                        WHERE (boards.user_id = ? OR shared_boards.user_id = ? OR boards.is_public = 1)
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
