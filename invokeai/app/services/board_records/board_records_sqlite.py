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
        self._conn = db.conn

    def delete(self, board_id: str) -> None:
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """--sql
                DELETE FROM boards
                WHERE board_id = ?;
                """,
                (board_id,),
            )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise BoardRecordDeleteException from e

    def save(
        self,
        board_name: str,
    ) -> BoardRecord:
        try:
            board_id = uuid_string()
            cursor = self._conn.cursor()
            cursor.execute(
                """--sql
                INSERT OR IGNORE INTO boards (board_id, board_name)
                VALUES (?, ?);
                """,
                (board_id, board_name),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        return self.get(board_id)

    def get(
        self,
        board_id: str,
    ) -> BoardRecord:
        try:
            cursor = self._conn.cursor()
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
        try:
            cursor = self._conn.cursor()
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

            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        return self.get(board_id)

    def get_many(
        self,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardRecord]:
        cursor = self._conn.cursor()

        # Build base query
        base_query = """
                SELECT *
                FROM boards
                {archived_filter}
                ORDER BY {order_by} {direction}
                LIMIT ? OFFSET ?;
            """

        # Determine archived filter condition
        archived_filter = "" if include_archived else "WHERE archived = 0"

        final_query = base_query.format(
            archived_filter=archived_filter, order_by=order_by.value, direction=direction.value
        )

        # Execute query to fetch boards
        cursor.execute(final_query, (limit, offset))

        result = cast(list[sqlite3.Row], cursor.fetchall())
        boards = [deserialize_board_record(dict(r)) for r in result]

        # Determine count query
        if include_archived:
            count_query = """
                    SELECT COUNT(*)
                    FROM boards;
                """
        else:
            count_query = """
                    SELECT COUNT(*)
                    FROM boards
                    WHERE archived = 0;
                """

        # Execute count query
        cursor.execute(count_query)

        count = cast(int, cursor.fetchone()[0])

        return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=count)

    def get_all(
        self, order_by: BoardRecordOrderBy, direction: SQLiteDirection, include_archived: bool = False
    ) -> list[BoardRecord]:
        cursor = self._conn.cursor()
        if order_by == BoardRecordOrderBy.Name:
            base_query = """
                    SELECT *
                    FROM boards
                    {archived_filter}
                    ORDER BY LOWER(board_name) {direction}
                """
        else:
            base_query = """
                    SELECT *
                    FROM boards
                    {archived_filter}
                    ORDER BY {order_by} {direction}
                """

        archived_filter = "" if include_archived else "WHERE archived = 0"

        final_query = base_query.format(
            archived_filter=archived_filter, order_by=order_by.value, direction=direction.value
        )

        cursor.execute(final_query)

        result = cast(list[sqlite3.Row], cursor.fetchall())
        boards = [deserialize_board_record(dict(r)) for r in result]

        return boards
