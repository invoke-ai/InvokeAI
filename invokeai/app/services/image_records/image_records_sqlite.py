import sqlite3
from datetime import datetime
from typing import Optional, Union, cast

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    IMAGE_DTO_COLS,
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ResourceOrigin,
    deserialize_image_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteImageRecordStorage(ImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, image_name: str) -> ImageRecord:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    f"""--sql
                    SELECT {IMAGE_DTO_COLS} FROM images
                    WHERE image_name = ?;
                    """,
                    (image_name,),
                )

                result = cast(Optional[sqlite3.Row], cursor.fetchone())
            except sqlite3.Error as e:
                raise ImageRecordNotFoundException from e

        if not result:
            raise ImageRecordNotFoundException

        return deserialize_image_record(dict(result))

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT metadata FROM images
                    WHERE image_name = ?;
                    """,
                    (image_name,),
                )

                result = cast(Optional[sqlite3.Row], cursor.fetchone())

            except sqlite3.Error as e:
                raise ImageRecordNotFoundException from e

            if not result:
                raise ImageRecordNotFoundException

            as_dict = dict(result)
            metadata_raw = cast(Optional[str], as_dict.get("metadata", None))
            return MetadataFieldValidator.validate_json(metadata_raw) if metadata_raw is not None else None

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> None:
        with self._db.transaction() as cursor:
            try:
                # Change the category of the image
                if changes.image_category is not None:
                    cursor.execute(
                        """--sql
                        UPDATE images
                        SET image_category = ?
                        WHERE image_name = ?;
                        """,
                        (changes.image_category, image_name),
                    )

                # Change the session associated with the image
                if changes.session_id is not None:
                    cursor.execute(
                        """--sql
                        UPDATE images
                        SET session_id = ?
                        WHERE image_name = ?;
                        """,
                        (changes.session_id, image_name),
                    )

                # Change the image's `is_intermediate`` flag
                if changes.is_intermediate is not None:
                    cursor.execute(
                        """--sql
                        UPDATE images
                        SET is_intermediate = ?
                        WHERE image_name = ?;
                        """,
                        (changes.is_intermediate, image_name),
                    )

                # Change the image's `starred`` state
                if changes.starred is not None:
                    cursor.execute(
                        """--sql
                        UPDATE images
                        SET starred = ?
                        WHERE image_name = ?;
                        """,
                        (changes.starred, image_name),
                    )

            except sqlite3.Error as e:
                raise ImageRecordSaveException from e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageRecord]:
        with self._db.transaction() as cursor:
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

            # Search term condition
            if search_term:
                query_conditions += """--sql
                AND (
                    images.metadata LIKE ?
                    OR images.created_at LIKE ?
                )
                """
                query_params.append(f"%{search_term.lower()}%")
                query_params.append(f"%{search_term.lower()}%")

            if starred_first:
                query_pagination = f"""--sql
                ORDER BY images.starred DESC, images.created_at {order_dir.value} LIMIT ? OFFSET ?
                """
            else:
                query_pagination = f"""--sql
                ORDER BY images.created_at {order_dir.value} LIMIT ? OFFSET ?
                """

            # Final images query with pagination
            images_query += query_conditions + query_pagination + ";"
            # Add all the parameters
            images_params = query_params.copy()
            # Add the pagination parameters
            images_params.extend([limit, offset])

            # Build the list of images, deserializing each row
            cursor.execute(images_query, images_params)
            result = cast(list[sqlite3.Row], cursor.fetchall())

            images = [deserialize_image_record(dict(r)) for r in result]

            # Set up and execute the count query, without pagination
            count_query += query_conditions + ";"
            count_params = query_params.copy()
            cursor.execute(count_query, count_params)
            count = cast(int, cursor.fetchone()[0])

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)

    def delete(self, image_name: str) -> None:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    DELETE FROM images
                    WHERE image_name = ?;
                    """,
                    (image_name,),
                )
            except sqlite3.Error as e:
                raise ImageRecordDeleteException from e

    def delete_many(self, image_names: list[str]) -> None:
        with self._db.transaction() as cursor:
            try:
                placeholders = ",".join("?" for _ in image_names)

                # Construct the SQLite query with the placeholders
                query = f"DELETE FROM images WHERE image_name IN ({placeholders})"

                # Execute the query with the list of IDs as parameters
                cursor.execute(query, image_names)

            except sqlite3.Error as e:
                raise ImageRecordDeleteException from e

    def get_intermediates_count(self) -> int:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT COUNT(*) FROM images
                WHERE is_intermediate = TRUE;
                """
            )
            count = cast(int, cursor.fetchone()[0])
        return count

    def delete_intermediates(self) -> list[str]:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT image_name FROM images
                    WHERE is_intermediate = TRUE;
                    """
                )
                result = cast(list[sqlite3.Row], cursor.fetchall())
                image_names = [r[0] for r in result]
                cursor.execute(
                    """--sql
                    DELETE FROM images
                    WHERE is_intermediate = TRUE;
                    """
                )
            except sqlite3.Error as e:
                raise ImageRecordDeleteException from e
        return image_names

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
        metadata: Optional[str] = None,
    ) -> datetime:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
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
                        metadata,
                        is_intermediate,
                        starred,
                        has_workflow,
                    ),
                )

                cursor.execute(
                    """--sql
                    SELECT created_at
                    FROM images
                    WHERE image_name = ?;
                    """,
                    (image_name,),
                )

                created_at = datetime.fromisoformat(cursor.fetchone()[0])

            except sqlite3.Error as e:
                raise ImageRecordSaveException from e
        return created_at

    def get_most_recent_image_for_board(self, board_id: str) -> Optional[ImageRecord]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT images.*
                FROM images
                JOIN board_images ON images.image_name = board_images.image_name
                WHERE board_images.board_id = ?
                AND images.is_intermediate = FALSE
                ORDER BY images.starred DESC, images.created_at DESC
                LIMIT 1;
                """,
                (board_id,),
            )

            result = cast(Optional[sqlite3.Row], cursor.fetchone())

        if result is None:
            return None

        return deserialize_image_record(dict(result))

    def get_image_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> ImageNamesResult:
        with self._db.transaction() as cursor:
            # Build query conditions (reused for both starred count and image names queries)
            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            if image_origin is not None:
                query_conditions += """--sql
                AND images.image_origin = ?
                """
                query_params.append(image_origin.value)

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                placeholders = ",".join("?" * len(category_strings))
                query_conditions += f"""--sql
                AND images.image_category IN ( {placeholders} )
                """
                for c in category_strings:
                    query_params.append(c)

            if is_intermediate is not None:
                query_conditions += """--sql
                AND images.is_intermediate = ?
                """
                query_params.append(is_intermediate)

            if board_id == "none":
                query_conditions += """--sql
                AND board_images.board_id IS NULL
                """
            elif board_id is not None:
                query_conditions += """--sql
                AND board_images.board_id = ?
                """
                query_params.append(board_id)

            if search_term:
                query_conditions += """--sql
                AND (
                    images.metadata LIKE ?
                    OR images.created_at LIKE ?
                )
                """
                query_params.append(f"%{search_term.lower()}%")
                query_params.append(f"%{search_term.lower()}%")

            # Get starred count if starred_first is enabled
            starred_count = 0
            if starred_first:
                starred_count_query = f"""--sql
                SELECT COUNT(*)
                FROM images
                LEFT JOIN board_images ON board_images.image_name = images.image_name
                WHERE images.starred = TRUE AND (1=1{query_conditions})
                """
                cursor.execute(starred_count_query, query_params)
                starred_count = cast(int, cursor.fetchone()[0])

            # Get all image names with proper ordering
            if starred_first:
                names_query = f"""--sql
                SELECT images.image_name
                FROM images
                LEFT JOIN board_images ON board_images.image_name = images.image_name
                WHERE 1=1{query_conditions}
                ORDER BY images.starred DESC, images.created_at {order_dir.value}
                """
            else:
                names_query = f"""--sql
                SELECT images.image_name
                FROM images
                LEFT JOIN board_images ON board_images.image_name = images.image_name
                WHERE 1=1{query_conditions}
                ORDER BY images.created_at {order_dir.value}
                """

            cursor.execute(names_query, query_params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
        image_names = [row[0] for row in result]

        return ImageNamesResult(image_names=image_names, starred_count=starred_count, total_count=len(image_names))
