import sqlite3
from dataclasses import dataclass
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
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO


@dataclass(frozen=True)
class _ImageQueryFilters:
    image_origin: Optional[ResourceOrigin] = None
    categories: Optional[list[ImageCategory]] = None
    is_intermediate: Optional[bool] = None
    board_id: Optional[str] = None
    search_term: Optional[str] = None
    created_from: Optional[str] = None
    created_to: Optional[str] = None
    user_id: Optional[str] = None
    is_admin: bool = False


# Every uncategorized image plus every image on an active (non-archived) board:
# the board_id="all" visibility scope for admins and single-user installs.
_ALL_ACTIVE_BOARDS_CONDITION = """(
                    board_images.board_id IS NULL
                    OR EXISTS (
                        SELECT 1
                        FROM boards
                        WHERE boards.board_id = board_images.board_id
                        AND boards.archived = 0
                    )
                )"""


def _build_image_query_conditions(filters: _ImageQueryFilters) -> tuple[str, list[Union[int, str, bool]]]:
    """Build the shared filters for image-list and image-name queries."""
    conditions: list[str] = []
    params: list[Union[int, str, bool]] = []

    if filters.image_origin is not None:
        conditions.append("images.image_origin = ?")
        params.append(filters.image_origin.value)

    if filters.categories is not None:
        category_strings = [category.value for category in set(filters.categories)]
        placeholders = ",".join("?" * len(category_strings))
        conditions.append(f"images.image_category IN ( {placeholders} )")
        params.extend(category_strings)

    if filters.is_intermediate is not None:
        conditions.append("images.is_intermediate = ?")
        params.append(filters.is_intermediate)

    if filters.board_id == "none":
        conditions.append("board_images.board_id IS NULL")
        if filters.user_id is not None and not filters.is_admin:
            conditions.append("images.user_id = ?")
            params.append(filters.user_id)
    elif filters.board_id == "all":
        if filters.is_admin:
            conditions.append(_ALL_ACTIVE_BOARDS_CONDITION)
        elif filters.user_id is not None:
            conditions.append(
                """(
                    (board_images.board_id IS NULL AND images.user_id = ?)
                    OR EXISTS (
                        SELECT 1
                        FROM boards
                        WHERE boards.board_id = board_images.board_id
                        AND boards.archived = 0
                        AND (
                            boards.user_id = ?
                            OR boards.board_visibility IN ('shared', 'public')
                            OR EXISTS (
                                SELECT 1
                                FROM shared_boards
                                WHERE shared_boards.board_id = boards.board_id
                                AND shared_boards.user_id = ?
                            )
                        )
                    )
                )"""
            )
            params.extend([filters.user_id, filters.user_id, filters.user_id])
        else:
            # Single-user mode has no current user; it reads the administrative scope.
            conditions.append(_ALL_ACTIVE_BOARDS_CONDITION)
    elif filters.board_id is not None:
        conditions.append("board_images.board_id = ?")
        params.append(filters.board_id)
    elif filters.user_id is not None and not filters.is_admin:
        conditions.append("images.user_id = ?")
        params.append(filters.user_id)

    if filters.search_term:
        conditions.append("(images.metadata LIKE ? OR images.created_at LIKE ?)")
        search_pattern = f"%{filters.search_term.lower()}%"
        params.extend([search_pattern, search_pattern])

    if filters.created_from:
        conditions.append("images.created_at >= ?")
        params.append(filters.created_from)

    if filters.created_to:
        conditions.append("images.created_at < DATE(?, '+1 day')")
        params.append(filters.created_to)

    return "".join(f"\nAND {condition}" for condition in conditions), params


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

    def get_user_id(self, image_name: str) -> Optional[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT user_id FROM images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            result = cast(Optional[sqlite3.Row], cursor.fetchone())
            if not result:
                return None
            return cast(Optional[str], dict(result).get("user_id"))

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

    def exists(self, image_name: str) -> bool:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT 1 FROM images
                WHERE image_name = ?
                LIMIT 1;
                """,
                (image_name,),
            )
            return cursor.fetchone() is not None

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
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
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

            query_conditions, query_params = _build_image_query_conditions(
                _ImageQueryFilters(
                    image_origin=image_origin,
                    categories=categories,
                    is_intermediate=is_intermediate,
                    board_id=board_id,
                    search_term=search_term,
                    created_from=created_from,
                    created_to=created_to,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            )

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

    def get_intermediates_count(self, user_id: Optional[str] = None) -> int:
        with self._db.transaction() as cursor:
            query = "SELECT COUNT(*) FROM images WHERE is_intermediate = TRUE"
            params: list[str] = []
            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor.execute(query, params)
            count = cast(int, cursor.fetchone()[0])
        return count

    def delete_intermediates(self) -> list[tuple[str, str]]:
        """Deletes all intermediate image records.

        Returns a list of (image_name, image_subfolder) tuples for file cleanup.
        """
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT image_name, image_subfolder FROM images
                    WHERE is_intermediate = TRUE;
                    """
                )
                result = cast(list[sqlite3.Row], cursor.fetchall())
                image_name_subfolder_pairs = [(r[0], r[1]) for r in result]
                cursor.execute(
                    """--sql
                    DELETE FROM images
                    WHERE is_intermediate = TRUE;
                    """
                )
            except sqlite3.Error as e:
                raise ImageRecordDeleteException from e
        return image_name_subfolder_pairs

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
        user_id: Optional[str] = None,
        image_subfolder: str = "",
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
                        has_workflow,
                        user_id,
                        image_subfolder
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                        user_id or "system",
                        image_subfolder,
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
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        with self._db.transaction() as cursor:
            query_conditions, query_params = _build_image_query_conditions(
                _ImageQueryFilters(
                    image_origin=image_origin,
                    categories=categories,
                    is_intermediate=is_intermediate,
                    board_id=board_id,
                    search_term=search_term,
                    created_from=created_from,
                    created_to=created_to,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            )

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

    def get_image_dates(
        self,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> list[VirtualSubBoardDTO]:
        with self._db.transaction() as cursor:
            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            # Only non-intermediate images
            query_conditions += """--sql
            AND images.is_intermediate = 0
            """

            # User isolation for non-admin users
            if user_id is not None and not is_admin:
                query_conditions += """--sql
                AND images.user_id = ?
                """
                query_params.append(user_id)

            query = f"""--sql
            SELECT
                DATE(images.created_at) as date,
                SUM(CASE WHEN images.image_category = 'general' THEN 1 ELSE 0 END) as image_count,
                SUM(CASE WHEN images.image_category != 'general' THEN 1 ELSE 0 END) as asset_count,
                (
                    SELECT i2.image_name FROM images i2
                    WHERE DATE(i2.created_at) = DATE(images.created_at)
                    AND i2.is_intermediate = 0
                    ORDER BY i2.created_at DESC LIMIT 1
                ) as cover_image_name
            FROM images
            WHERE 1=1
            {query_conditions}
            GROUP BY DATE(images.created_at)
            ORDER BY date DESC;
            """

            cursor.execute(query, query_params)
            result = cast(list[sqlite3.Row], cursor.fetchall())

        return [
            VirtualSubBoardDTO(
                virtual_board_id=f"by_date:{dict(row)['date']}",
                board_name=dict(row)["date"],
                date=dict(row)["date"],
                image_count=dict(row)["image_count"],
                asset_count=dict(row)["asset_count"],
                cover_image_name=dict(row)["cover_image_name"],
            )
            for row in result
        ]

    def get_image_names_by_date(
        self,
        date: str,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        categories: Optional[list[ImageCategory]] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        with self._db.transaction() as cursor:
            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            # Filter by date
            query_conditions += """--sql
            AND DATE(images.created_at) = ?
            """
            query_params.append(date)

            # Only non-intermediate images
            query_conditions += """--sql
            AND images.is_intermediate = 0
            """

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                placeholders = ",".join("?" * len(category_strings))
                query_conditions += f"""--sql
                AND images.image_category IN ( {placeholders} )
                """
                for c in category_strings:
                    query_params.append(c)

            # User isolation for non-admin users
            if user_id is not None and not is_admin:
                query_conditions += """--sql
                AND images.user_id = ?
                """
                query_params.append(user_id)

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
                WHERE images.starred = TRUE AND (1=1{query_conditions})
                """
                cursor.execute(starred_count_query, query_params)
                starred_count = cast(int, cursor.fetchone()[0])

            # Get all image names with proper ordering
            if starred_first:
                names_query = f"""--sql
                SELECT images.image_name
                FROM images
                WHERE 1=1{query_conditions}
                ORDER BY images.starred DESC, images.created_at {order_dir.value}
                """
            else:
                names_query = f"""--sql
                SELECT images.image_name
                FROM images
                WHERE 1=1{query_conditions}
                ORDER BY images.created_at {order_dir.value}
                """

            cursor.execute(names_query, query_params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
        image_names = [row[0] for row in result]

        return ImageNamesResult(image_names=image_names, starred_count=starred_count, total_count=len(image_names))
