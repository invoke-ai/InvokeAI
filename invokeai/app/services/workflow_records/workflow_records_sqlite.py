from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite.sqlite_migrator import Migration, MigrationSet
from invokeai.app.services.workflow_records.migrations import v0, v1
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
)

workflows_migrations = MigrationSet(
    table_name="workflows",
    migrations=[
        Migration(version=0, migrate=v0),
        Migration(version=1, migrate=v1),
    ],
)


class SqliteWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()
        self._db.register_migration_set(workflows_migrations)

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT workflow_id, workflow, created_at, updated_at
                FROM workflows
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

    def create(self, workflow: Workflow) -> WorkflowRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO workflows(workflow)
                VALUES (?);
                """,
                (workflow.model_dump_json(),),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(workflow.id)

    def update(self, workflow: Workflow) -> WorkflowRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE workflows
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
                DELETE from workflows
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

    def get_many(self, page: int, per_page: int) -> PaginatedResults[WorkflowRecordDTO]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                """--sql
                SELECT workflow_id, workflow, created_at, updated_at
                FROM workflows
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
            """,
                (per_page, page * per_page),
            )
            rows = self._cursor.fetchall()
            workflows = [WorkflowRecordDTO.from_dict(dict(row)) for row in rows]
            self._cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM workflows;
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
