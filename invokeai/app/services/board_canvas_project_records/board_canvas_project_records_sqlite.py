import sqlite3
from typing import Optional, cast

from invokeai.app.services.board_canvas_project_records.board_canvas_project_records_base import (
    BoardCanvasProjectRecordStorageBase,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteBoardCanvasProjectRecordStorage(BoardCanvasProjectRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def add_project_to_board(self, board_id: str, project_name: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT INTO board_canvas_projects (board_id, project_name)
                VALUES (?, ?)
                ON CONFLICT (project_name) DO UPDATE SET board_id = ?;
                """,
                (board_id, project_name, board_id),
            )

    def remove_project_from_board(self, project_name: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM board_canvas_projects WHERE project_name = ?;",
                (project_name,),
            )

    def get_all_board_project_names_for_board(
        self,
        board_id: str,
        is_intermediate: Optional[bool] = None,
    ) -> list[str]:
        with self._db.transaction() as cursor:
            params: list[str | bool] = []
            stmt = """
                SELECT canvas_projects.project_name
                FROM canvas_projects
                LEFT JOIN board_canvas_projects
                    ON board_canvas_projects.project_name = canvas_projects.project_name
                WHERE 1=1
                """
            if board_id == "none":
                stmt += " AND board_canvas_projects.board_id IS NULL "
            else:
                stmt += " AND board_canvas_projects.board_id = ? "
                params.append(board_id)

            if is_intermediate is not None:
                stmt += " AND canvas_projects.is_intermediate = ? "
                params.append(is_intermediate)

            stmt += ";"
            cursor.execute(stmt, params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
        return [r[0] for r in result]

    def get_board_for_project(self, project_name: str) -> Optional[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                "SELECT board_id FROM board_canvas_projects WHERE project_name = ?;",
                (project_name,),
            )
            result = cursor.fetchone()
        if result is None:
            return None
        return cast(str, result[0])

    def get_project_count_for_board(self, board_id: str) -> int:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM board_canvas_projects
                INNER JOIN canvas_projects
                    ON board_canvas_projects.project_name = canvas_projects.project_name
                WHERE canvas_projects.is_intermediate = FALSE
                AND board_canvas_projects.board_id = ?;
                """,
                (board_id,),
            )
            count = cast(int, cursor.fetchone()[0])
        return count
