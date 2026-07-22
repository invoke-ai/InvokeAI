import sqlite3
from typing import Optional, cast

from invokeai.app.services.board_video_records.board_video_records_base import BoardVideoRecordStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteBoardVideoRecordStorage(BoardVideoRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def add_video_to_board(self, board_id: str, video_name: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT INTO board_videos (board_id, video_name)
                VALUES (?, ?)
                ON CONFLICT (video_name) DO UPDATE SET board_id = ?;
                """,
                (board_id, video_name, board_id),
            )

    def remove_video_from_board(self, video_name: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM board_videos WHERE video_name = ?;",
                (video_name,),
            )

    def get_all_board_video_names_for_board(
        self,
        board_id: str,
        categories: list[ImageCategory] | None,
        is_intermediate: bool | None,
        user_id: Optional[str] = None,
    ) -> list[str]:
        with self._db.transaction() as cursor:
            params: list[str | bool] = []
            stmt = """
                SELECT videos.video_name
                FROM videos
                LEFT JOIN board_videos ON board_videos.video_name = videos.video_name
                WHERE 1=1
                """
            if board_id == "none":
                stmt += " AND board_videos.board_id IS NULL "
            else:
                stmt += " AND board_videos.board_id = ? "
                params.append(board_id)

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                placeholders = ",".join("?" * len(category_strings))
                stmt += f" AND videos.video_category IN ( {placeholders} ) "
                for c in category_strings:
                    params.append(c)

            if is_intermediate is not None:
                stmt += " AND videos.is_intermediate = ? "
                params.append(is_intermediate)

            if user_id is not None:
                stmt += " AND videos.user_id = ? "
                params.append(user_id)

            stmt += ";"
            cursor.execute(stmt, params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
        return [r[0] for r in result]

    def get_board_for_video(self, video_name: str) -> Optional[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                "SELECT board_id FROM board_videos WHERE video_name = ?;",
                (video_name,),
            )
            result = cursor.fetchone()
        if result is None:
            return None
        return cast(str, result[0])

    def get_video_count_for_board(self, board_id: str) -> int:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM board_videos
                INNER JOIN videos ON board_videos.video_name = videos.video_name
                WHERE videos.is_intermediate = FALSE
                AND board_videos.board_id = ?;
                """,
                (board_id,),
            )
            count = cast(int, cursor.fetchone()[0])
        return count
