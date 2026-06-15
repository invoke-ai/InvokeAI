import sqlite3
from datetime import datetime
from typing import Optional, Union, cast

from invokeai.app.services.canvas_project_records.canvas_project_records_base import CanvasProjectRecordStorageBase
from invokeai.app.services.canvas_project_records.canvas_project_records_common import (
    CANVAS_PROJECT_DTO_COLS,
    CanvasProjectNamesResult,
    CanvasProjectRecord,
    CanvasProjectRecordChanges,
    CanvasProjectRecordDeleteException,
    CanvasProjectRecordNotFoundException,
    CanvasProjectRecordSaveException,
    deserialize_canvas_project_record,
)
from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteCanvasProjectRecordStorage(CanvasProjectRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, project_name: str) -> CanvasProjectRecord:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    f"""--sql
                    SELECT {CANVAS_PROJECT_DTO_COLS} FROM canvas_projects
                    WHERE project_name = ?;
                    """,
                    (project_name,),
                )
                result = cast(Optional[sqlite3.Row], cursor.fetchone())
            except sqlite3.Error as e:
                raise CanvasProjectRecordNotFoundException from e

        if not result:
            raise CanvasProjectRecordNotFoundException
        return deserialize_canvas_project_record(dict(result))

    def get_user_id(self, project_name: str) -> Optional[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT user_id FROM canvas_projects
                WHERE project_name = ?;
                """,
                (project_name,),
            )
            result = cast(Optional[sqlite3.Row], cursor.fetchone())
            if not result:
                return None
            return cast(Optional[str], dict(result).get("user_id"))

    def update(self, project_name: str, changes: CanvasProjectRecordChanges) -> None:
        with self._db.transaction() as cursor:
            try:
                if changes.name is not None:
                    cursor.execute(
                        "UPDATE canvas_projects SET name = ? WHERE project_name = ?;",
                        (changes.name, project_name),
                    )
                if changes.is_intermediate is not None:
                    cursor.execute(
                        "UPDATE canvas_projects SET is_intermediate = ? WHERE project_name = ?;",
                        (changes.is_intermediate, project_name),
                    )
                if changes.starred is not None:
                    cursor.execute(
                        "UPDATE canvas_projects SET starred = ? WHERE project_name = ?;",
                        (changes.starred, project_name),
                    )
            except sqlite3.Error as e:
                raise CanvasProjectRecordSaveException from e

    def set_has_thumbnail(self, project_name: str, has_thumbnail: bool) -> None:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    "UPDATE canvas_projects SET has_thumbnail = ? WHERE project_name = ?;",
                    (has_thumbnail, project_name),
                )
            except sqlite3.Error as e:
                raise CanvasProjectRecordSaveException from e

    def update_file_metadata(
        self,
        project_name: str,
        width: int,
        height: int,
        image_count: int,
        has_thumbnail: bool,
        app_version: str,
    ) -> None:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    UPDATE canvas_projects
                    SET width = ?, height = ?, image_count = ?, has_thumbnail = ?, app_version = ?
                    WHERE project_name = ?;
                    """,
                    (width, height, image_count, has_thumbnail, app_version, project_name),
                )
            except sqlite3.Error as e:
                raise CanvasProjectRecordSaveException from e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        project_origin: Optional[ResourceOrigin] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[CanvasProjectRecord]:
        with self._db.transaction() as cursor:
            count_query = """--sql
            SELECT COUNT(*)
            FROM canvas_projects
            LEFT JOIN board_canvas_projects ON board_canvas_projects.project_name = canvas_projects.project_name
            WHERE 1=1
            """
            projects_query = f"""--sql
            SELECT {CANVAS_PROJECT_DTO_COLS}
            FROM canvas_projects
            LEFT JOIN board_canvas_projects ON board_canvas_projects.project_name = canvas_projects.project_name
            WHERE 1=1
            """

            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            if project_origin is not None:
                query_conditions += " AND canvas_projects.project_origin = ? "
                query_params.append(project_origin.value)

            if is_intermediate is not None:
                query_conditions += " AND canvas_projects.is_intermediate = ? "
                query_params.append(is_intermediate)

            if board_id == "none":
                query_conditions += " AND board_canvas_projects.board_id IS NULL "
                if user_id is not None and not is_admin:
                    query_conditions += " AND canvas_projects.user_id = ? "
                    query_params.append(user_id)
            elif board_id is not None:
                query_conditions += " AND board_canvas_projects.board_id = ? "
                query_params.append(board_id)
            elif user_id is not None and not is_admin:
                query_conditions += " AND canvas_projects.user_id = ? "
                query_params.append(user_id)

            if search_term:
                query_conditions += " AND (canvas_projects.name LIKE ? OR canvas_projects.created_at LIKE ?) "
                query_params.append(f"%{search_term.lower()}%")
                query_params.append(f"%{search_term.lower()}%")

            if starred_first:
                query_pagination = (
                    f" ORDER BY canvas_projects.starred DESC, canvas_projects.created_at {order_dir.value} "
                    "LIMIT ? OFFSET ? "
                )
            else:
                query_pagination = f" ORDER BY canvas_projects.created_at {order_dir.value} LIMIT ? OFFSET ? "

            projects_query += query_conditions + query_pagination + ";"
            projects_params = query_params.copy()
            projects_params.extend([limit, offset])
            cursor.execute(projects_query, projects_params)
            result = cast(list[sqlite3.Row], cursor.fetchall())
            projects = [deserialize_canvas_project_record(dict(r)) for r in result]

            count_query += query_conditions + ";"
            cursor.execute(count_query, query_params.copy())
            count = cast(int, cursor.fetchone()[0])

        return OffsetPaginatedResults(items=projects, offset=offset, limit=limit, total=count)

    def delete(self, project_name: str) -> None:
        with self._db.transaction() as cursor:
            try:
                cursor.execute("DELETE FROM canvas_projects WHERE project_name = ?;", (project_name,))
            except sqlite3.Error as e:
                raise CanvasProjectRecordDeleteException from e

    def delete_many(self, project_names: list[str]) -> None:
        with self._db.transaction() as cursor:
            try:
                placeholders = ",".join("?" for _ in project_names)
                cursor.execute(
                    f"DELETE FROM canvas_projects WHERE project_name IN ({placeholders})", project_names
                )
            except sqlite3.Error as e:
                raise CanvasProjectRecordDeleteException from e

    def save(
        self,
        project_name: str,
        project_origin: ResourceOrigin,
        name: str,
        app_version: str,
        width: int,
        height: int,
        image_count: int,
        has_thumbnail: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        user_id: Optional[str] = None,
        project_subfolder: str = "",
    ) -> datetime:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO canvas_projects (
                        project_name,
                        project_origin,
                        name,
                        app_version,
                        width,
                        height,
                        image_count,
                        has_thumbnail,
                        is_intermediate,
                        starred,
                        user_id,
                        project_subfolder
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        project_name,
                        project_origin.value,
                        name,
                        app_version,
                        width,
                        height,
                        image_count,
                        has_thumbnail,
                        is_intermediate,
                        starred,
                        user_id or "system",
                        project_subfolder,
                    ),
                )

                cursor.execute(
                    "SELECT created_at FROM canvas_projects WHERE project_name = ?;",
                    (project_name,),
                )
                created_at = datetime.fromisoformat(cursor.fetchone()[0])
            except sqlite3.Error as e:
                raise CanvasProjectRecordSaveException from e
        return created_at

    def get_project_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        project_origin: Optional[ResourceOrigin] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> CanvasProjectNamesResult:
        with self._db.transaction() as cursor:
            query_conditions = ""
            query_params: list[Union[int, str, bool]] = []

            if project_origin is not None:
                query_conditions += " AND canvas_projects.project_origin = ? "
                query_params.append(project_origin.value)

            if is_intermediate is not None:
                query_conditions += " AND canvas_projects.is_intermediate = ? "
                query_params.append(is_intermediate)

            if board_id == "none":
                query_conditions += " AND board_canvas_projects.board_id IS NULL "
                if user_id is not None and not is_admin:
                    query_conditions += " AND canvas_projects.user_id = ? "
                    query_params.append(user_id)
            elif board_id is not None:
                query_conditions += " AND board_canvas_projects.board_id = ? "
                query_params.append(board_id)
            elif user_id is not None and not is_admin:
                query_conditions += " AND canvas_projects.user_id = ? "
                query_params.append(user_id)

            if search_term:
                query_conditions += " AND (canvas_projects.name LIKE ? OR canvas_projects.created_at LIKE ?) "
                query_params.append(f"%{search_term.lower()}%")
                query_params.append(f"%{search_term.lower()}%")

            starred_count = 0
            if starred_first:
                cursor.execute(
                    f"""--sql
                    SELECT COUNT(*)
                    FROM canvas_projects
                    LEFT JOIN board_canvas_projects
                        ON board_canvas_projects.project_name = canvas_projects.project_name
                    WHERE canvas_projects.starred = TRUE AND (1=1{query_conditions})
                    """,
                    query_params,
                )
                starred_count = cast(int, cursor.fetchone()[0])

            order_clause = (
                f" ORDER BY canvas_projects.starred DESC, canvas_projects.created_at {order_dir.value} "
                if starred_first
                else f" ORDER BY canvas_projects.created_at {order_dir.value} "
            )
            cursor.execute(
                f"""--sql
                SELECT canvas_projects.project_name
                FROM canvas_projects
                LEFT JOIN board_canvas_projects
                    ON board_canvas_projects.project_name = canvas_projects.project_name
                WHERE 1=1{query_conditions}
                {order_clause}
                """,
                query_params,
            )
            result = cast(list[sqlite3.Row], cursor.fetchall())
        project_names = [row[0] for row in result]
        return CanvasProjectNamesResult(
            project_names=project_names, starred_count=starred_count, total_count=len(project_names)
        )
