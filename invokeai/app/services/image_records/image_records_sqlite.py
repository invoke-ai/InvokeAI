import sqlite3
import threading
from datetime import datetime
from typing import Optional, Union, cast

from invokeai.app.invocations.baseinvocation import MetadataField, MetadataFieldValidator
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

from .image_records_base import ImageRecordStorageBase
from .image_records_common import (
    IMAGE_DTO_COLS,
    ImageCategory,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ResourceOrigin,
    deserialize_image_record,
)


class SqliteImageRecordStorage(ImageRecordStorageBase):
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.RLock

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()

    def get(self, image_name: str) -> ImageRecord:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT {IMAGE_DTO_COLS} FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )

            result = cast(Optional[sqlite3.Row], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordNotFoundException from e
        finally:
            self._lock.release()

        if not result:
            raise ImageRecordNotFoundException

        return deserialize_image_record(dict(result))

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                """--sql
                SELECT metadata FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )

            result = cast(Optional[sqlite3.Row], self._cursor.fetchone())

            if not result:
                raise ImageRecordNotFoundException

            as_dict = dict(result)
            metadata_raw = cast(Optional[str], as_dict.get("metadata", None))
            return MetadataFieldValidator.validate_json(metadata_raw) if metadata_raw is not None else None
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordNotFoundException from e
        finally:
            self._lock.release()

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> None:
        try:
            self._lock.acquire()
            # Change the category of the image
            if changes.image_category is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE images
                    SET image_category = ?
                    WHERE image_name = ?;
                    """,
                    (changes.image_category, image_name),
                )

            # Change the session associated with the image
            if changes.session_id is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE images
                    SET session_id = ?
                    WHERE image_name = ?;
                    """,
                    (changes.session_id, image_name),
                )

            # Change the image's `is_intermediate`` flag
            if changes.is_intermediate is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE images
                    SET is_intermediate = ?
                    WHERE image_name = ?;
                    """,
                    (changes.is_intermediate, image_name),
                )

            # Change the image's `starred`` state
            if changes.starred is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE images
                    SET starred = ?
                    WHERE image_name = ?;
                    """,
                    (changes.starred, image_name),
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
        board_id: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageRecord]:
        try:
            self._lock.acquire()

            # Manually build two queries - one for the count, one for the records
            count_query = """--sql
            SELECT COUNT(*)
            FROM images
            LEFT JOIN board_images ON board_images.image_name = images.image_name
            WHERE 1=1
            """

            images_query = f"""--sql
            SELECT {IMAGE_DTO_COLS}
            FROM images
            LEFT JOIN board_images ON board_images.image_name = images.image_name
            WHERE 1=1
            """

            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            if image_origin is not None:
                query_conditions += """--sql
                AND images.image_origin = ?
                """
                query_params.append(image_origin.value)

            if categories is not None:
                # Convert the enum values to unique list of strings
                category_strings = [c.value for c in set(categories)]
                # Create the correct length of placeholders
                placeholders = ",".join("?" * len(category_strings))

                query_conditions += f"""--sql
                AND images.image_category IN ( {placeholders} )
                """

                # Unpack the included categories into the query params
                for c in category_strings:
                    query_params.append(c)

            if is_intermediate is not None:
                query_conditions += """--sql
                AND images.is_intermediate = ?
                """

                query_params.append(is_intermediate)

            # board_id of "none" is reserved for images without a board
            if board_id == "none":
                query_conditions += """--sql
                AND board_images.board_id IS NULL
                """
            elif board_id is not None:
                query_conditions += """--sql
                AND board_images.board_id = ?
                """
                query_params.append(board_id)

            query_pagination = """--sql
            ORDER BY images.starred DESC, images.created_at DESC LIMIT ? OFFSET ?
            """

            # Final images query with pagination
            images_query += query_conditions + query_pagination + ";"
            # Add all the parameters
            images_params = query_params.copy()
            # Add the pagination parameters
            images_params.extend([limit, offset])

            # Build the list of images, deserializing each row
            self._cursor.execute(images_query, images_params)
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            images = [deserialize_image_record(dict(r)) for r in result]

            # Set up and execute the count query, without pagination
            count_query += query_conditions + ";"
            count_params = query_params.copy()
            self._cursor.execute(count_query, count_params)
            count = cast(int, self._cursor.fetchone()[0])
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)

    def delete(self, image_name: str) -> None:
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

    def delete_many(self, image_names: list[str]) -> None:
        try:
            placeholders = ",".join("?" for _ in image_names)

            self._lock.acquire()

            # Construct the SQLite query with the placeholders
            query = f"DELETE FROM images WHERE image_name IN ({placeholders})"

            # Execute the query with the list of IDs as parameters
            self._cursor.execute(query, image_names)

            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordDeleteException from e
        finally:
            self._lock.release()

    def get_intermediates_count(self) -> int:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT COUNT(*) FROM images
                WHERE is_intermediate = TRUE;
                """
            )
            count = cast(int, self._cursor.fetchone()[0])
            self._conn.commit()
            return count
        except sqlite3.Error as e:
            self._conn.rollback()
            raise ImageRecordDeleteException from e
        finally:
            self._lock.release()

    def delete_intermediates(self) -> list[str]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT image_name FROM images
                WHERE is_intermediate = TRUE;
                """
            )
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            image_names = [r[0] for r in result]
            self._cursor.execute(
                """--sql
                DELETE FROM images
                WHERE is_intermediate = TRUE;
                """
            )
            self._conn.commit()
            return image_names
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
        width: int,
        height: int,
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[MetadataField] = None,
    ) -> datetime:
        try:
            metadata_json = metadata.model_dump_json() if metadata is not None else None
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
                    is_intermediate,
                    starred,
                    has_workflow
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    starred,
                    has_workflow,
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

    def get_most_recent_image_for_board(self, board_id: str) -> Optional[ImageRecord]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT images.*
                FROM images
                JOIN board_images ON images.image_name = board_images.image_name
                WHERE board_images.board_id = ?
                ORDER BY images.starred DESC, images.created_at DESC
                LIMIT 1;
                """,
                (board_id,),
            )

            result = cast(Optional[sqlite3.Row], self._cursor.fetchone())
        finally:
            self._lock.release()
        if result is None:
            return None

        return deserialize_image_record(dict(result))