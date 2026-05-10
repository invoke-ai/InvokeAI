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

    def create(self, system_prompt: SystemPromptWithoutId) -> SystemPromptRecordDTO:
        system_prompt_id = uuid_string()
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT INTO system_prompts (id, name, content)
                VALUES (?, ?, ?);
                """,
                (system_prompt_id, system_prompt.name, system_prompt.content),
            )
        return self.get(system_prompt_id)

    def update(self, system_prompt_id: str, changes: SystemPromptChanges) -> SystemPromptRecordDTO:
        with self._db.transaction() as cursor:
            # Ensure the record exists so we can raise a clean 404 instead of silently no-op'ing.
            cursor.execute("SELECT 1 FROM system_prompts WHERE id = ?;", (system_prompt_id,))
            if cursor.fetchone() is None:
                raise SystemPromptNotFoundError(f"System prompt with id {system_prompt_id} not found")

            if changes.name is not None:
                cursor.execute(
                    "UPDATE system_prompts SET name = ? WHERE id = ?;",
                    (changes.name, system_prompt_id),
                )
            if changes.content is not None:
                cursor.execute(
                    "UPDATE system_prompts SET content = ? WHERE id = ?;",
                    (changes.content, system_prompt_id),
                )
        return self.get(system_prompt_id)

    def delete(self, system_prompt_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM system_prompts WHERE id = ?;",
                (system_prompt_id,),
            )

    def get_many(self) -> list[SystemPromptRecordDTO]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT * FROM system_prompts ORDER BY LOWER(name) ASC;
                """
            )
            rows = cursor.fetchall()
        return [SystemPromptRecordDTO.from_dict(dict(row)) for row in rows]
