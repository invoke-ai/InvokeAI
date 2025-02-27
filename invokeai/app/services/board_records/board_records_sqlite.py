import sqlite3
import threading
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
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.RLock

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()

    def delete(self, board_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM boards
                WHERE board_id = ?;
                """,
                (board_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordDeleteException from e
        except Exception as e:
            self._conn.rollback()
            raise BoardRecordDeleteException from e
        finally:
            self._lock.release()

    def save(
        self,
        board_name: str,
    ) -> BoardRecord:
        try:
            board_id = uuid_string()
            self._lock.acquire()
            self._cursor.execute(
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
        finally:
            self._lock.release()
        return self.get(board_id)

    def get(
        self,
        board_id: str,
    ) -> BoardRecord:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM boards
                WHERE board_id = ?;
                """,
                (board_id,),
            )

            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordNotFoundException from e
        finally:
            self._lock.release()
        if result is None:
            raise BoardRecordNotFoundException
        return BoardRecord(**dict(result))

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardRecord:
        try:
            self._lock.acquire()

            # Change the name of a board
            if changes.board_name is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE boards
                    SET board_name = ?
                    WHERE board_id = ?;
                    """,
                    (changes.board_name, board_id),
                )

            # Change the cover image of a board
            if changes.cover_image_name is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE boards
                    SET cover_image_name = ?
                    WHERE board_id = ?;
                    """,
                    (changes.cover_image_name, board_id),
                )

            # Change the archived status of a board
            if changes.archived is not None:
                self._cursor.execute(
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
        finally:
            self._lock.release()
        return self.get(board_id)

    def get_many(
        self,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardRecord]:
        try:
            self._lock.acquire()

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
            self._cursor.execute(final_query, (limit, offset))

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
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
            self._cursor.execute(count_query)

            count = cast(int, self._cursor.fetchone()[0])

            return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=count)

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_all(
        self, order_by: BoardRecordOrderBy, direction: SQLiteDirection, include_archived: bool = False
    ) -> list[BoardRecord]:
        try:
            self._lock.acquire()

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

            self._cursor.execute(final_query)

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]

            return boards

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()
