from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, Optional, TypeVar, cast
import sqlite3
import threading
from typing import Optional, Union

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.models.image import (
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.services.models.image_record import (
    ImageRecord,
    ImageRecordChanges,
    deserialize_image_record,
)

T = TypeVar("T", bound=BaseModel)

class OffsetPaginatedResults(GenericModel, Generic[T]):
    """Offset-paginated results"""

    # fmt: off
    items: list[T] = Field(description="Items")
    offset: int = Field(description="Offset from which to retrieve items")
    limit: int = Field(description="Limit of items to get")
    total: int = Field(description="Total number of items in result")
    # fmt: on


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
    def get(self, image_origin: ResourceOrigin, image_name: str) -> ImageRecord:
        """Gets an image record."""
        pass

    @abstractmethod
    def update(
        self,
        image_name: str,
        image_origin: ResourceOrigin,
        changes: ImageRecordChanges,
    ) -> None:
        """Updates an image record."""
        pass

    @abstractmethod
    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> OffsetPaginatedResults[ImageRecord]:
        """Gets a page of image records."""
        pass

    # TODO: The database has a nullable `deleted_at` column, currently unused.
    # Should we implement soft deletes? Would need coordination with ImageFileStorage.
    @abstractmethod
    def delete(self, image_origin: ResourceOrigin, image_name: str) -> None:
        """Deletes an image record."""
        pass

    @abstractmethod
    def save(
        self,
        image_name: str,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        width: int,
        height: int,
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: Optional[ImageMetadata],
        is_intermediate: bool = False,
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
                image_origin TEXT NOT NULL,
                -- This is an enum in python, unrestricted string here for flexibility
                image_category TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                is_intermediate BOOLEAN DEFAULT FALSE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
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
            CREATE INDEX IF NOT EXISTS idx_images_image_origin ON images(image_origin);
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

    def get(
        self, image_origin: ResourceOrigin, image_name: str
    ) -> Union[ImageRecord, None]:
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

    def update(
        self,
        image_name: str,
        image_origin: ResourceOrigin,
        changes: ImageRecordChanges,
    ) -> None:
        try:
            self._lock.acquire()
            # Change the category of the image
            if changes.image_category is not None:
                self._cursor.execute(
                    f"""--sql
                    UPDATE images
                    SET image_category = ?
                    WHERE image_name = ?;
                    """,
                    (changes.image_category, image_name),
                )

            # Change the session associated with the image
            if changes.session_id is not None:
                self._cursor.execute(
                    f"""--sql
                    UPDATE images
                    SET session_id = ?
                    WHERE image_name = ?;
                    """,
                    (changes.session_id, image_name),
                )

            # Change the image's `is_intermediate`` flag
            if changes.is_intermediate is not None:
                self._cursor.execute(
                    f"""--sql
                    UPDATE images
                    SET is_intermediate = ?
                    WHERE image_name = ?;
                    """,
                    (changes.is_intermediate, image_name),
                )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordSaveException from e
        finally:
            self._lock.release()

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> OffsetPaginatedResults[ImageRecord]:
        try:
            self._lock.acquire()

            # Manually build two queries - one for the count, one for the records

            count_query = f"""SELECT COUNT(*) FROM images WHERE 1=1\n"""
            images_query = f"""SELECT * FROM images WHERE 1=1\n"""

            query_conditions = ""
            query_params = []

            if image_origin is not None:
                query_conditions += f"""AND image_origin = ?\n"""
                query_params.append(image_origin.value)

            if categories is not None:
                ## Convert the enum values to unique list of strings
                category_strings = list(
                    map(lambda c: c.value, set(categories))
                )
                # Create the correct length of placeholders
                placeholders = ",".join("?" * len(category_strings))
                query_conditions += f"AND image_category IN ( {placeholders} )\n"

                # Unpack the included categories into the query params
                for c in category_strings:
                    query_params.append(c)

            if is_intermediate is not None:
                query_conditions += f"""AND is_intermediate = ?\n"""
                query_params.append(is_intermediate)

            query_pagination = f"""ORDER BY created_at DESC LIMIT ? OFFSET ?\n"""

            # Final images query with pagination
            images_query += query_conditions + query_pagination + ";"
            # Add all the parameters
            images_params = query_params.copy()
            images_params.append(limit)
            images_params.append(offset)
            # Build the list of images, deserializing each row
            self._cursor.execute(images_query, images_params)
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            images = list(map(lambda r: deserialize_image_record(dict(r)), result))

            # Set up and execute the count query, without pagination
            count_query += query_conditions + ";"
            count_params = query_params.copy()
            self._cursor.execute(count_query, count_params)
            count = self._cursor.fetchone()[0]
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

        return OffsetPaginatedResults(
            items=images, offset=offset, limit=limit, total=count
        )

    def delete(self, image_origin: ResourceOrigin, image_name: str) -> None:
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
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        session_id: Optional[str],
        width: int,
        height: int,
        node_id: Optional[str],
        metadata: Optional[ImageMetadata],
        is_intermediate: bool = False,
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
                    image_origin,
                    image_category,
                    width,
                    height,
                    node_id,
                    session_id,
                    metadata,
                    is_intermediate
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    image_name,
                    image_origin.value,
                    image_category.value,
                    width,
                    height,
                    node_id,
                    session_id,
                    metadata_json,
                    is_intermediate,
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
