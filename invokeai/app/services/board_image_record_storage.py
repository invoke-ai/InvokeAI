from abc import ABC, abstractmethod
import sqlite3
import threading
from typing import Optional, cast

from invokeai.app.services.image_record_storage import OffsetPaginatedResults
from invokeai.app.services.models.image_record import (
    ImageRecord,
    deserialize_image_record,
)


class BoardImageRecordStorageBase(ABC):
    """Abstract base class for the one-to-many board-image relationship record storage."""

    @abstractmethod
    def add_image_to_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        """Adds an image to a board."""
        pass

    @abstractmethod
    def remove_image_from_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        """Removes an image from a board."""
        pass

    @abstractmethod
    def get_all_board_image_names_for_board(
        self,
        board_id: str,
    ) -> list[str]:
        """Gets all board images for a board, as a list of the image names."""
        pass

    @abstractmethod
    def get_board_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        """Gets an image's board id, if it has one."""
        pass

    @abstractmethod
    def get_image_count_for_board(
        self,
        board_id: str,
    ) -> int:
        """Gets the number of images for a board."""
        pass


class SqliteBoardImageRecordStorage(BoardImageRecordStorageBase):
    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, filename: str) -> None:
        super().__init__()
        self._filename = filename
        self._conn = sqlite3.connect(filename, check_same_thread=False)
        # Enable row factory to get rows as dictionaries (must be done before making the cursor!)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()

        try:
            self._lock.acquire()
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._create_tables()
            self._conn.commit()
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        """Creates the `board_images` junction table."""

        # Create the `board_images` junction table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS board_images (
                board_id TEXT NOT NULL,
                image_name TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                -- enforce one-to-many relationship between boards and images using PK
                -- (we can extend this to many-to-many later)
                PRIMARY KEY (image_name),
                FOREIGN KEY (board_id) REFERENCES boards (board_id) ON DELETE CASCADE,
                FOREIGN KEY (image_name) REFERENCES images (image_name) ON DELETE CASCADE
            );
            """
        )

        # Add index for board id
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_board_images_board_id ON board_images (board_id);
            """
        )

        # Add index for board id, sorted by created_at
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_board_images_board_id_created_at ON board_images (board_id, created_at);
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_board_images_updated_at
            AFTER UPDATE
            ON board_images FOR EACH ROW
            BEGIN
                UPDATE board_images SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE board_id = old.board_id AND image_name = old.image_name;
            END;
            """
        )

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
        board_id: str,
        image_name: str,
    ) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM board_images
                WHERE board_id = ? AND image_name = ?;
                """,
                (board_id, image_name),
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
            images = list(map(lambda r: deserialize_image_record(dict(r)), result))

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
            image_names = list(map(lambda r: r[0], result))
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
                SELECT COUNT(*) FROM board_images WHERE board_id = ?;
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
