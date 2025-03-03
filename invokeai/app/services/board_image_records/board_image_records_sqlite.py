import sqlite3
from typing import Optional, cast

from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecord,
    deserialize_image_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteBoardImageRecordStorage(BoardImageRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._conn = db.conn

    def add_image_to_board(
        self,
        board_id: str,
        image_name: str,
    ) -> None:
        try:
            cursor = self._conn.cursor()
            cursor.execute(
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

    def remove_image_from_board(
        self,
        image_name: str,
    ) -> None:
        try:
            cursor = self._conn.cursor()
            cursor.execute(
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

    def get_images_for_board(
        self,
        board_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[ImageRecord]:
        # TODO: this isn't paginated yet?
        cursor = self._conn.cursor()
        cursor.execute(
            """--sql
            SELECT images.*
            FROM board_images
            INNER JOIN images ON board_images.image_name = images.image_name
            WHERE board_images.board_id = ?
            ORDER BY board_images.updated_at DESC;
            """,
            (board_id,),
        )
        result = cast(list[sqlite3.Row], cursor.fetchall())
        images = [deserialize_image_record(dict(r)) for r in result]

        cursor.execute(
            """--sql
            SELECT COUNT(*) FROM images WHERE 1=1;
            """
        )
        count = cast(int, cursor.fetchone()[0])

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)

    def get_all_board_image_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
    ) -> list[str]:
        params: list[str | bool] = []

        # Base query is a join between images and board_images
        stmt = """
                SELECT images.image_name
                FROM images
                LEFT JOIN board_images ON board_images.image_name = images.image_name
                WHERE 1=1
                AND board_images.board_id = ?
                """
        params.append(board_id)

        # Add the category filter
        if categories is not None:
            # Convert the enum values to unique list of strings
            category_strings = [c.value for c in set(categories)]
            # Create the correct length of placeholders
            placeholders = ",".join("?" * len(category_strings))
            stmt += f"""--sql
                AND images.image_category IN ( {placeholders} )
                """

            # Unpack the included categories into the query params
            for c in category_strings:
                params.append(c)

        # Add the is_intermediate filter
        if is_intermediate is not None:
            stmt += """--sql
                AND images.is_intermediate = ?
                """
            params.append(is_intermediate)

        # Put a ring on it
        stmt += ";"

        # Execute the query
        cursor = self._conn.cursor()
        cursor.execute(stmt, params)

        result = cast(list[sqlite3.Row], cursor.fetchall())
        image_names = [r[0] for r in result]
        return image_names

    def get_board_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        cursor = self._conn.cursor()
        cursor.execute(
            """--sql
                SELECT board_id
                FROM board_images
                WHERE image_name = ?;
                """,
            (image_name,),
        )
        result = cursor.fetchone()
        if result is None:
            return None
        return cast(str, result[0])

    def get_image_count_for_board(self, board_id: str) -> int:
        cursor = self._conn.cursor()
        cursor.execute(
            """--sql
                SELECT COUNT(*)
                FROM board_images
                INNER JOIN images ON board_images.image_name = images.image_name
                WHERE images.is_intermediate = FALSE
                AND board_images.board_id = ?;
                """,
            (board_id,),
        )
        count = cast(int, cursor.fetchone()[0])
        return count
