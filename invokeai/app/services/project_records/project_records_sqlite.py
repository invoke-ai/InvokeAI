import json
import sqlite3
import uuid
from typing import Any

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.project_records.project_records_base import ProjectRecordsStorageBase
from invokeai.app.services.project_records.project_records_common import (
    ProjectRecordConflictError,
    ProjectRecordDTO,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
    ProjectSummaryDTO,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class ProjectRecordsSqlite(ProjectRecordsStorageBase):
    """SQLite implementation of per-user project document storage."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def create(self, user_id: str, name: str, data: dict[str, Any], project_id: str | None = None) -> ProjectRecordDTO:
        project_id = project_id or uuid.uuid4().hex

        try:
            with self._db.transaction() as cursor:
                cursor.execute(
                    """--sql
                    INSERT INTO projects (project_id, user_id, name, data)
                    VALUES (?, ?, ?, ?);
                    """,
                    (project_id, user_id, name, json.dumps(data)),
                )
        except sqlite3.IntegrityError as e:
            raise ProjectRecordExistsError(project_id) from e

        return self.get(user_id, project_id)

    def get(self, user_id: str, project_id: str) -> ProjectRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT project_id, name, data, revision, created_at, updated_at
                FROM projects
                WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
            row = cursor.fetchone()

        if row is None:
            raise ProjectRecordNotFoundError(project_id)

        return ProjectRecordDTO(
            project_id=row[0],
            name=row[1],
            data=json.loads(row[2]),
            revision=row[3],
            created_at=row[4],
            updated_at=row[5],
        )

    def list(self, user_id: str) -> list[ProjectSummaryDTO]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT project_id, name, revision, created_at, updated_at
                FROM projects
                WHERE user_id = ?
                -- rowid breaks ties between rows created in the same millisecond,
                -- keeping the listing in true insertion order.
                ORDER BY created_at ASC, rowid ASC;
                """,
                (user_id,),
            )
            rows = cursor.fetchall()

        return [
            ProjectSummaryDTO(
                project_id=row[0],
                name=row[1],
                revision=row[2],
                created_at=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]

    def update(
        self, user_id: str, project_id: str, expected_revision: int, name: str, data: dict[str, Any]
    ) -> ProjectRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE projects
                SET name = ?, data = ?, revision = revision + 1
                WHERE user_id = ? AND project_id = ? AND revision = ?;
                """,
                (name, json.dumps(data), user_id, project_id, expected_revision),
            )

            if cursor.rowcount == 0:
                # Distinguish "gone" from "someone else saved first".
                cursor.execute(
                    """--sql
                    SELECT revision FROM projects
                    WHERE user_id = ? AND project_id = ?;
                    """,
                    (user_id, project_id),
                )
                row = cursor.fetchone()

                if row is None:
                    raise ProjectRecordNotFoundError(project_id)

                raise ProjectRecordConflictError(project_id, expected_revision, row[0])

        return self.get(user_id, project_id)

    def delete(self, user_id: str, project_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM projects
                WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
