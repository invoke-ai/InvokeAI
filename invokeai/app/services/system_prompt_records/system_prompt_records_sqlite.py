from typing import Optional

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.system_prompt_records.system_prompt_records_base import (
    SystemPromptRecordsStorageBase,
)
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptChanges,
    SystemPromptNotFoundError,
    SystemPromptRecordDTO,
    SystemPromptWithoutId,
)
from invokeai.app.util.misc import uuid_string


class SqliteSystemPromptRecordsStorage(SystemPromptRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, system_prompt_id: str) -> SystemPromptRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT * FROM system_prompts WHERE id = ?;
                """,
                (system_prompt_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise SystemPromptNotFoundError(f"System prompt with id {system_prompt_id} not found")
        return SystemPromptRecordDTO.from_dict(dict(row))

    def create(
        self,
        system_prompt: SystemPromptWithoutId,
        user_id: str,
        is_public: bool = False,
    ) -> SystemPromptRecordDTO:
        system_prompt_id = uuid_string()
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT INTO system_prompts (id, name, content, user_id, is_public)
                VALUES (?, ?, ?, ?, ?);
                """,
                (system_prompt_id, system_prompt.name, system_prompt.content, user_id, is_public),
            )
        return self.get(system_prompt_id)

    def update(
        self,
        system_prompt_id: str,
        changes: SystemPromptChanges,
        user_id: Optional[str] = None,
    ) -> SystemPromptRecordDTO:
        with self._db.transaction() as cursor:
            # Confirm the row exists and (if scoped) is owned by the caller — distinguishes 404 from 403.
            if user_id is not None:
                cursor.execute(
                    "SELECT 1 FROM system_prompts WHERE id = ? AND user_id = ?;",
                    (system_prompt_id, user_id),
                )
            else:
                cursor.execute("SELECT 1 FROM system_prompts WHERE id = ?;", (system_prompt_id,))
            if cursor.fetchone() is None:
                raise SystemPromptNotFoundError(f"System prompt with id {system_prompt_id} not found")

            scope_clause = " AND user_id = ?" if user_id is not None else ""
            scope_args: tuple = (user_id,) if user_id is not None else ()

            if changes.name is not None:
                cursor.execute(
                    f"UPDATE system_prompts SET name = ? WHERE id = ?{scope_clause};",
                    (changes.name, system_prompt_id, *scope_args),
                )
            if changes.content is not None:
                cursor.execute(
                    f"UPDATE system_prompts SET content = ? WHERE id = ?{scope_clause};",
                    (changes.content, system_prompt_id, *scope_args),
                )
            if changes.is_public is not None:
                cursor.execute(
                    f"UPDATE system_prompts SET is_public = ? WHERE id = ?{scope_clause};",
                    (changes.is_public, system_prompt_id, *scope_args),
                )
        return self.get(system_prompt_id)

    def delete(self, system_prompt_id: str, user_id: Optional[str] = None) -> None:
        with self._db.transaction() as cursor:
            if user_id is not None:
                cursor.execute(
                    "DELETE FROM system_prompts WHERE id = ? AND user_id = ?;",
                    (system_prompt_id, user_id),
                )
            else:
                cursor.execute("DELETE FROM system_prompts WHERE id = ?;", (system_prompt_id,))

    def get_many(self, user_id: Optional[str] = None) -> list[SystemPromptRecordDTO]:
        with self._db.transaction() as cursor:
            if user_id is not None:
                cursor.execute(
                    """--sql
                    SELECT * FROM system_prompts
                    WHERE user_id = ? OR is_public = TRUE
                    ORDER BY LOWER(name) ASC;
                    """,
                    (user_id,),
                )
            else:
                cursor.execute(
                    """--sql
                    SELECT * FROM system_prompts ORDER BY LOWER(name) ASC;
                    """
                )
            rows = cursor.fetchall()
        return [SystemPromptRecordDTO.from_dict(dict(row)) for row in rows]
