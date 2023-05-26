from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, cast
import sqlite3
import threading
from typing import Optional, Union

from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
)
from invokeai.app.services.models.image_record import (
    ImageRecord,
    deserialize_image_record,
)
from invokeai.app.services.item_storage import PaginatedResults


# TODO: Should these excpetions subclass existing python exceptions?
class ImageRecordNotFoundException(Exception):
    """Raised when an image record is not found."""

    def __init__(self, message="Image record not found"):
        super().__init__(message)


class ImageRecordSaveException(Exception):
    """Raised when an image record cannot be saved."""

    def __init__(self, message="Image record not saved"):
        super().__init__(message)


class ImageRecordDeleteException(Exception):
    """Raised when an image record cannot be deleted."""

    def __init__(self, message="Image record not deleted"):
        super().__init__(message)


class ImageRecordStorageBase(ABC):
    """Low-level service responsible for interfacing with the image record store."""

    # TODO: Implement an `update()` method

    @abstractmethod
    def get(self, image_type: ImageType, image_name: str) -> ImageRecord:
        """Gets an image record."""
        pass

    @abstractmethod
    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageRecord]:
        """Gets a page of image records."""
        pass

    # TODO: The database has a nullable `deleted_at` column, currently unused.
    # Should we implement soft deletes? Would need coordination with ImageFileStorage.
    @abstractmethod
    def delete(self, image_type: ImageType, image_name: str) -> None:
        """Deletes an image record."""
        pass

    @abstractmethod
    def save(
        self,
        image_name: str,
        image_type: ImageType,
        image_category: ImageCategory,
        width: int,
        height: int,
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: Optional[ImageMetadata],
    ) -> datetime:
        """Saves an image record."""
        pass


class SqliteImageRecordStorage(ImageRecordStorageBase):
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
        """Creates the tables for the `images` database."""

        # Create the `images` table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS images (
                image_name TEXT NOT NULL PRIMARY KEY,
                -- This is an enum in python, unrestricted string here for flexibility
                image_type TEXT NOT NULL,
                -- This is an enum in python, unrestricted string here for flexibility
                image_category TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                -- Soft delete, currently unused
                deleted_at DATETIME
            );
            """
        )

        # Create the `images` table indices.
        self._cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_images_image_name ON images(image_name);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_image_type ON images(image_type);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_image_category ON images(image_category);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_images_updated_at
            AFTER UPDATE
            ON images FOR EACH ROW
            BEGIN
                UPDATE images SET updated_at = current_timestamp
                    WHERE image_name = old.image_name;
            END;
            """
        )

    def get(self, image_type: ImageType, image_name: str) -> Union[ImageRecord, None]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT * FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )

            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordNotFoundException from e
        finally:
            self._lock.release()

        if not result:
            raise ImageRecordNotFoundException

        return deserialize_image_record(dict(result))

    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageRecord]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT * FROM images
                WHERE image_type = ? AND image_category = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
                """,
                (image_type.value, image_category.value, per_page, page * per_page),
            )

            result = cast(list[sqlite3.Row], self._cursor.fetchall())

            images = list(map(lambda r: deserialize_image_record(dict(r)), result))

            self._cursor.execute(
                """--sql
                SELECT count(*) FROM images
                WHERE image_type = ? AND image_category = ?
                """,
                (image_type.value, image_category.value),
            )

            count = self._cursor.fetchone()[0]
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults(
            items=images, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def delete(self, image_type: ImageType, image_name: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordDeleteException from e
        finally:
            self._lock.release()

    def save(
        self,
        image_name: str,
        image_type: ImageType,
        image_category: ImageCategory,
        session_id: Optional[str],
        width: int,
        height: int,
        node_id: Optional[str],
        metadata: Optional[ImageMetadata],
    ) -> datetime:
        try:
            metadata_json = (
                None if metadata is None else metadata.json(exclude_none=True)
            )
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO images (
                    image_name,
                    image_type,
                    image_category,
                    width,
                    height,
                    node_id,
                    session_id,
                    metadata
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    image_name,
                    image_type.value,
                    image_category.value,
                    width,
                    height,
                    node_id,
                    session_id,
                    metadata_json,
                ),
            )
            self._conn.commit()

            self._cursor.execute(
                """--sql
                SELECT created_at
                FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )

            created_at = datetime.fromisoformat(self._cursor.fetchone()[0])

            return created_at
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordSaveException from e
        finally:
            self._lock.release()
