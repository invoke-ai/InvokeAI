import sqlite3
from datetime import datetime
from typing import Optional, Union, cast

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.video_records.video_records_base import VideoRecordStorageBase
from invokeai.app.services.video_records.video_records_common import (
    VIDEO_DTO_COLS,
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
    VideoRecordDeleteException,
    VideoRecordNotFoundException,
    VideoRecordSaveException,
    deserialize_video_record,
)


class SqliteVideoRecordStorage(VideoRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, video_name: str) -> VideoRecord:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    f"""--sql
                    SELECT {VIDEO_DTO_COLS} FROM videos
                    WHERE video_name = ?;
                    """,
                    (video_name,),
                )
                result = cast(Optional[sqlite3.Row], cursor.fetchone())
            except sqlite3.Error as e:
                raise VideoRecordNotFoundException from e

        if not result:
            raise VideoRecordNotFoundException
        return deserialize_video_record(dict(result))

    def get_user_id(self, video_name: str) -> Optional[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT user_id FROM videos
                WHERE video_name = ?;
                """,
                (video_name,),
            )
            result = cast(Optional[sqlite3.Row], cursor.fetchone())
            if not result:
                return None
            return cast(Optional[str], dict(result).get("user_id"))

    def get_most_recent_video_for_board(self, board_id: str) -> Optional[VideoRecord]:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT {VIDEO_DTO_COLS}
                FROM videos
                JOIN board_videos ON videos.video_name = board_videos.video_name
                WHERE board_videos.board_id = ?
                AND videos.is_intermediate = FALSE
                ORDER BY videos.starred DESC, videos.created_at DESC, videos.video_name DESC
                LIMIT 1;
                """,
                (board_id,),
            )
            result = cast(Optional[sqlite3.Row], cursor.fetchone())
        if result is None:
            return None
        return deserialize_video_record(dict(result))

    def get_metadata(self, video_name: str) -> Optional[MetadataField]:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT metadata FROM videos
                    WHERE video_name = ?;
                    """,
                    (video_name,),
                )
                result = cast(Optional[sqlite3.Row], cursor.fetchone())
            except sqlite3.Error as e:
                raise VideoRecordNotFoundException from e

            if not result:
                raise VideoRecordNotFoundException

            as_dict = dict(result)
            metadata_raw = cast(Optional[str], as_dict.get("metadata", None))
            return MetadataFieldValidator.validate_json(metadata_raw) if metadata_raw is not None else None

    def update(self, video_name: str, changes: VideoRecordChanges) -> None:
        with self._db.transaction() as cursor:
            try:
                if changes.video_category is not None:
                    cursor.execute(
                        "UPDATE videos SET video_category = ? WHERE video_name = ?;",
                        (changes.video_category.value, video_name),
                    )
                if changes.session_id is not None:
                    cursor.execute(
                        "UPDATE videos SET session_id = ? WHERE video_name = ?;",
                        (changes.session_id, video_name),
                    )
                if changes.is_intermediate is not None:
                    cursor.execute(
                        "UPDATE videos SET is_intermediate = ? WHERE video_name = ?;",
                        (changes.is_intermediate, video_name),
                    )
                if changes.starred is not None:
                    cursor.execute(
                        "UPDATE videos SET starred = ? WHERE video_name = ?;",
                        (changes.starred, video_name),
                    )
            except sqlite3.Error as e:
                raise VideoRecordSaveException from e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        video_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[VideoRecord]:
        with self._db.transaction() as cursor:
            count_query = """--sql
            SELECT COUNT(*)
            FROM videos
            LEFT JOIN board_videos ON board_videos.video_name = videos.video_name
            WHERE 1=1
            """
            videos_query = f"""--sql
            SELECT {VIDEO_DTO_COLS}
            FROM videos
            LEFT JOIN board_videos ON board_videos.video_name = videos.video_name
            WHERE 1=1
            """

            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            if video_origin is not None:
                query_conditions += " AND videos.video_origin = ? "
                query_params.append(video_origin.value)

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                placeholders = ",".join("?" * len(category_strings))
                query_conditions += f" AND videos.video_category IN ( {placeholders} ) "
                for c in category_strings:
                    query_params.append(c)

            if is_intermediate is not None:
                query_conditions += " AND videos.is_intermediate = ? "
                query_params.append(is_intermediate)

            if board_id == "none":
                query_conditions += " AND board_videos.board_id IS NULL "
                if user_id is not None and not is_admin:
                    query_conditions += " AND videos.user_id = ? "
                    query_params.append(user_id)
            elif board_id is not None:
                query_conditions += " AND board_videos.board_id = ? "
                query_params.append(board_id)
            elif user_id is not None and not is_admin:
                # No board_id supplied — still enforce per-user isolation so
                # non-admin callers cannot enumerate other users' videos.
                query_conditions += " AND videos.user_id = ? "
                query_params.append(user_id)

            if search_term:
                query_conditions += " AND (videos.metadata LIKE ? OR videos.created_at LIKE ?) "
                query_params.append(f"%{search_term.lower()}%")
                query_params.append(f"%{search_term.lower()}%")

            if starred_first:
                query_pagination = (
                    f" ORDER BY videos.starred DESC, videos.created_at {order_dir.value}, "
                    f"videos.video_name {order_dir.value} LIMIT ? OFFSET ? "
                )
            else:
                query_pagination = (
                    f" ORDER BY videos.created_at {order_dir.value}, videos.video_name {order_dir.value} "
                    "LIMIT ? OFFSET ? "
                )

            videos_query += query_conditions + query_pagination + ";"
            videos_params = query_params.copy()
            videos_params.extend([limit, offset])
            cursor.execute(videos_query, videos_params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
            videos = [deserialize_video_record(dict(r)) for r in result]

            count_query += query_conditions + ";"
            cursor.execute(count_query, query_params.copy())
            count = cast(int, cursor.fetchone()[0])

        return OffsetPaginatedResults(items=videos, offset=offset, limit=limit, total=count)

    def delete(self, video_name: str) -> None:
        with self._db.transaction() as cursor:
            try:
                cursor.execute("DELETE FROM videos WHERE video_name = ?;", (video_name,))
            except sqlite3.Error as e:
                raise VideoRecordDeleteException from e

    def delete_many(self, video_names: list[str]) -> None:
        with self._db.transaction() as cursor:
            try:
                placeholders = ",".join("?" for _ in video_names)
                cursor.execute(f"DELETE FROM videos WHERE video_name IN ({placeholders})", video_names)
            except sqlite3.Error as e:
                raise VideoRecordDeleteException from e

    def save(
        self,
        video_name: str,
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        width: int,
        height: int,
        duration: float,
        fps: Optional[float],
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[str] = None,
        user_id: Optional[str] = None,
        video_subfolder: str = "",
    ) -> datetime:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO videos (
                        video_name,
                        video_origin,
                        video_category,
                        width,
                        height,
                        duration,
                        fps,
                        node_id,
                        session_id,
                        metadata,
                        is_intermediate,
                        starred,
                        has_workflow,
                        user_id,
                        video_subfolder
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        video_name,
                        video_origin.value,
                        video_category.value,
                        width,
                        height,
                        float(duration),
                        float(fps) if fps is not None else None,
                        node_id,
                        session_id,
                        metadata,
                        is_intermediate,
                        starred,
                        has_workflow,
                        user_id or "system",
                        video_subfolder,
                    ),
                )

                cursor.execute(
                    "SELECT created_at FROM videos WHERE video_name = ?;",
                    (video_name,),
                )
                created_at = datetime.fromisoformat(cursor.fetchone()[0])
            except sqlite3.Error as e:
                raise VideoRecordSaveException from e
        return created_at

    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        video_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> VideoNamesResult:
        with self._db.transaction() as cursor:
            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            if video_origin is not None:
                query_conditions += " AND videos.video_origin = ? "
                query_params.append(video_origin.value)

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                placeholders = ",".join("?" * len(category_strings))
                query_conditions += f" AND videos.video_category IN ( {placeholders} ) "
                for c in category_strings:
                    query_params.append(c)

            if is_intermediate is not None:
                query_conditions += " AND videos.is_intermediate = ? "
                query_params.append(is_intermediate)

            if board_id == "none":
                query_conditions += " AND board_videos.board_id IS NULL "
                if user_id is not None and not is_admin:
                    query_conditions += " AND videos.user_id = ? "
                    query_params.append(user_id)
            elif board_id is not None:
                query_conditions += " AND board_videos.board_id = ? "
                query_params.append(board_id)
            elif user_id is not None and not is_admin:
                # No board_id supplied — still enforce per-user isolation so
                # non-admin callers cannot enumerate other users' videos.
                query_conditions += " AND videos.user_id = ? "
                query_params.append(user_id)

            if search_term:
                query_conditions += " AND (videos.metadata LIKE ? OR videos.created_at LIKE ?) "
                query_params.append(f"%{search_term.lower()}%")
                query_params.append(f"%{search_term.lower()}%")

            starred_count = 0
            if starred_first:
                cursor.execute(
                    f"""--sql
                    SELECT COUNT(*)
                    FROM videos
                    LEFT JOIN board_videos ON board_videos.video_name = videos.video_name
                    WHERE videos.starred = TRUE AND (1=1{query_conditions})
                    """,
                    query_params,
                )
                starred_count = cast(int, cursor.fetchone()[0])

            order_clause = (
                f" ORDER BY videos.starred DESC, videos.created_at {order_dir.value}, "
                f"videos.video_name {order_dir.value} "
                if starred_first
                else f" ORDER BY videos.created_at {order_dir.value}, videos.video_name {order_dir.value} "
            )
            cursor.execute(
                f"""--sql
                SELECT videos.video_name
                FROM videos
                LEFT JOIN board_videos ON board_videos.video_name = videos.video_name
                WHERE 1=1{query_conditions}
                {order_clause}
                """,
                query_params,
            )
            result = cast(list[sqlite3.Row], cursor.fetchall())
        video_names = [row[0] for row in result]
        return VideoNamesResult(video_names=video_names, starred_count=starred_count, total_count=len(video_names))
