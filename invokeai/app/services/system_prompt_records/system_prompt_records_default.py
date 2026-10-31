from typing import Optional

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
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


def _not_found(system_prompt_id: str) -> SystemPromptNotFoundError:
    return SystemPromptNotFoundError(f"System prompt with id {system_prompt_id} not found")


def _found(prompt: Optional[SystemPromptRecordDTO], system_prompt_id: str) -> SystemPromptRecordDTO:
    if prompt is None:
        raise _not_found(system_prompt_id)
    return prompt


class SystemPromptRecordsStorage(SystemPromptRecordsStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def get(self, system_prompt_id: str) -> SystemPromptRecordDTO:
        return _found(self._queries.system_prompts.get(system_prompt_id), system_prompt_id)

    def create(
        self,
        system_prompt: SystemPromptWithoutId,
        user_id: str,
        is_public: bool = False,
    ) -> SystemPromptRecordDTO:
        system_prompt_id = uuid_string()
        created_at = self._queries.system_prompts.insert(system_prompt_id, system_prompt, user_id, is_public)
        # The record is what was stored, so it is not read back.
        return SystemPromptRecordDTO(
            **system_prompt.model_dump(),
            id=system_prompt_id,
            user_id=user_id,
            is_public=is_public,
            created_at=created_at,
            updated_at=created_at,
        )

    def update(
        self,
        system_prompt_id: str,
        changes: SystemPromptChanges,
        user_id: Optional[str] = None,
    ) -> SystemPromptRecordDTO:
        def apply(q: Queries) -> Optional[SystemPromptRecordDTO]:
            # A prompt that does not exist and, when scoped, one the account does not own are alike not found.
            if not q.system_prompts.update(system_prompt_id, changes, user_id):
                raise _not_found(system_prompt_id)
            return q.system_prompts.get(system_prompt_id)

        return _found(self._queries.run(apply), system_prompt_id)

    def delete(self, system_prompt_id: str, user_id: Optional[str] = None) -> None:
        """Delete a prompt, optionally scoped to an owner.

        Raises `SystemPromptNotFoundError` when nothing was deleted -- either the id does not
        exist or (when scoped) it is not owned by `user_id`. Deleting silently would let the
        single-user router report success for an id that `GET` 404s on, and would make the
        symmetry with `update()` (which does raise) a trap for the next caller.
        """
        if not self._queries.system_prompts.delete(system_prompt_id, user_id):
            raise _not_found(system_prompt_id)

    def get_many(self, user_id: Optional[str] = None) -> list[SystemPromptRecordDTO]:
        return self._queries.system_prompts.all(user_id)
