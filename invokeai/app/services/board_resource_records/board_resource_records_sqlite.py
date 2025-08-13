import sqlite3
from typing import Optional, cast

from invokeai.app.services.board_resource_records.board_resource_records_base import BoardResourceRecordStorageBase

from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.resources.resources_common import ResourceIdentifier, ResourceType
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteBoardResourceRecordStorage(BoardResourceRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def add_resource_to_board(
        self,
        board_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT INTO board_images (board_id, image_name)
                VALUES (?, ?)
                ON CONFLICT (image_name) DO UPDATE SET board_id = ?;
                """,
                (board_id, resource_id, board_id),
            )

    def remove_resource_from_board(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM board_images
                WHERE image_name = ?;
                """,
                (resource_id,),
            )


    def get_all_board_resource_ids_for_board(
        self,
        board_id: str,
        resource_type: Optional[ResourceType] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> list[ResourceIdentifier]:
        image_name_results = []

        with self._db.transaction() as cursor:
            if resource_type == ResourceType.IMAGE or resource_type is None:
                params: list[str | bool] = []

                # Base query is a join between images and board_images
                stmt = """
                        SELECT images.image_name
                        FROM images
                        LEFT JOIN board_images ON board_images.image_name = images.image_name
                        WHERE 1=1
                        """

                # Handle board_id filter
                if board_id == "none":
                    stmt += """--sql
                        AND board_images.board_id IS NULL
                        """
                else:
                    stmt += """--sql
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

                cursor.execute(stmt, params)

                image_name_results = cast(list[sqlite3.Row], cursor.fetchall())

            if resource_type == ResourceType.VIDEO or resource_type is None:
            #    this is not actually a valid code path for OSS, just demonstrating that it could be
                raise NotImplementedError("Video resource type is not supported in OSS")

            return [ResourceIdentifier(resource_id=image_name, resource_type=ResourceType.IMAGE) for image_name in image_name_results]

    def get_board_for_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> Optional[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                    SELECT board_id
                    FROM board_images
                    WHERE image_name = ?;
                    """,
                (resource_id,),
            )
            result = cursor.fetchone()
        if result is None:
            return None
        return cast(str, result[0])

    def get_resource_count_for_board(self, board_id: str, resource_type: Optional[ResourceType] = None) -> int:
        with self._db.transaction() as cursor:
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
