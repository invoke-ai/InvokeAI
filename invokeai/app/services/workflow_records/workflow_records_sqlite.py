from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordListItemDTOValidator,
    WorkflowValidator,
    WorkflowWithoutID,
)


class SqliteWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()
        self._create_tables()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT workflow_id, workflow, created_at, updated_at
                FROM workflow_library
                WHERE workflow_id = ?;
                """,
                (workflow_id,),
            )
            row = self._cursor.fetchone()
            if row is None:
                raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
            return WorkflowRecordDTO.from_dict(dict(row))
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def create(self, workflow: WorkflowWithoutID) -> WorkflowRecordDTO:
        try:
            workflow_with_id = WorkflowValidator.validate_python(workflow.model_dump())
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO workflow_library (
                    workflow_id,
                    workflow
                )
                VALUES (?, ?);
                """,
                (
                    workflow_with_id.id,
                    workflow_with_id.model_dump_json(),
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(workflow_with_id.id)

    def update(self, workflow: Workflow) -> WorkflowRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?
                WHERE workflow_id = ?;
                """,
                (workflow.model_dump_json(), workflow.id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(workflow.id)

    def delete(self, workflow_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE from workflow_library
                WHERE workflow_id = ?;
                """,
                (workflow_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return None

    def get_many(self, page: int, per_page: int) -> PaginatedResults[WorkflowRecordListItemDTO]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                """--sql
                SELECT
                    workflow_id,
                    json_extract(workflow, '$.name') AS name,
                    json_extract(workflow, '$.description') AS description,
                    created_at,
                    updated_at
                FROM workflow_library
                ORDER BY name ASC
                LIMIT ? OFFSET ?;
                """,
                (per_page, page * per_page),
            )
            rows = self._cursor.fetchall()
            workflows = [WorkflowRecordListItemDTOValidator.validate_python(dict(row)) for row in rows]
            self._cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM workflow_library;
                """
            )
            total = self._cursor.fetchone()[0]
            pages = int(total / per_page) + 1
            return PaginatedResults(
                items=workflows,
                page=page,
                per_page=per_page,
                pages=pages,
                total=total,
            )
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                CREATE TABLE IF NOT EXISTS workflow_library (
                    workflow_id TEXT NOT NULL PRIMARY KEY, -- gets implicit index
                    workflow TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                    updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')) -- updated via trigger
                );
                """
            )

            self._cursor.execute(
                """--sql
                CREATE TRIGGER IF NOT EXISTS tg_workflow_library_updated_at
                AFTER UPDATE
                ON workflow_library FOR EACH ROW
                BEGIN
                    UPDATE workflow_library
                    SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE workflow_id = old.workflow_id;
                END;
                """
            )

            # We do not need the original `workflows` table or `workflow_images` junction table.
            self._cursor.execute(
                """--sql
                DROP TABLE IF EXISTS workflow_images;
                """
            )
            self._cursor.execute(
                """--sql
                DROP TABLE IF EXISTS workflows;
                """
            )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
