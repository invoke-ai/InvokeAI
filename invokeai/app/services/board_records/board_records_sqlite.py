import sqlite3
import threading
from typing import Union, cast

from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.misc import uuid_string

from .board_records_base import BoardRecordStorageBase
from .board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordDeleteException,
    BoardRecordNotFoundException,
    BoardRecordSaveException,
    deserialize_board_record,
)


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

            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        finally:
            self._lock.release()
        return self.get(board_id)

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[BoardRecord]:
        try:
            self._lock.acquire()

            # Get all the boards
            self._cursor.execute(
                """--sql
                SELECT *
                FROM boards
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
                """,
                (limit, offset),
            )

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]

            # Get the total number of boards
            self._cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM boards
                WHERE 1=1;
                """
            )

            count = cast(int, self._cursor.fetchone()[0])

            return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=count)

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_all(
        self,
    ) -> list[BoardRecord]:
        try:
            self._lock.acquire()

            # Get all the boards
            self._cursor.execute(
                """--sql
                SELECT *
                FROM boards
                ORDER BY created_at DESC
                """
            )

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]

            return boards

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()
