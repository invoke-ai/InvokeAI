import sqlite3
import threading
from typing import Optional, cast

from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import ImageRecord, deserialize_image_record
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteBoardImageRecordStorage(BoardImageRecordStorageBase):
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.RLock

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()

    def add_image_to_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT INTO board_images (board_id, image_name)
                VALUES (?, ?)
                ON CONFLICT (image_name) DO UPDATE SET board_id = ?;
                """,
                (board_id, image_name, board_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def remove_image_from_board(
        self,
        image_name: str,
    ) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM board_images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_images_for_board(
        self,
        board_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[ImageRecord]:
        # TODO: this isn't paginated yet?
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT images.*
                FROM board_images
                INNER JOIN images ON board_images.image_name = images.image_name
                WHERE board_images.board_id = ?
                ORDER BY board_images.updated_at DESC;
                """,
                (board_id,),
            )
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            images = [deserialize_image_record(dict(r)) for r in result]

            self._cursor.execute(
                """--sql
                SELECT COUNT(*) FROM images WHERE 1=1;
                """
            )
            count = cast(int, self._cursor.fetchone()[0])

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()
        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)

    def get_all_board_image_names_for_board(self, board_id: str) -> list[str]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT image_name
                FROM board_images
                WHERE board_id = ?;
                """,
                (board_id,),
            )
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            image_names = [r[0] for r in result]
            return image_names
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_board_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT board_id
                FROM board_images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            result = self._cursor.fetchone()
            if result is None:
                return None
            return cast(str, result[0])
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_image_count_for_board(self, board_id: str) -> int:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM board_images
                INNER JOIN images ON board_images.image_name = images.image_name
                WHERE images.is_intermediate = FALSE
                AND board_images.board_id = ?;
                """,
                (board_id,),
            )
            count = cast(int, self._cursor.fetchone()[0])
            return count
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()
