from typing import Optional

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import UniqueViolation
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.wildcard_records.wildcard_records_base import WildcardRecordsStorageBase
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardRecordDTO,
    WildcardWithoutId,
)
from invokeai.app.util.misc import uuid_string


def _found(wildcard: Optional[WildcardRecordDTO], wildcard_id: str) -> WildcardRecordDTO:
    if wildcard is None:
        raise WildcardNotFoundError(f"Wildcard with id {wildcard_id} not found")
    return wildcard


def _name_taken(name: Optional[str]) -> WildcardNameConflictError:
    return WildcardNameConflictError(f"A wildcard named '{name}' already exists")


class WildcardRecordsStorage(WildcardRecordsStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def get(self, wildcard_id: str) -> WildcardRecordDTO:
        return _found(self._queries.wildcards.get(wildcard_id), wildcard_id)

    def create(self, wildcard: WildcardWithoutId, user_id: str) -> WildcardRecordDTO:
        wildcard_id = uuid_string()
        try:
            self._queries.wildcards.insert(wildcard_id, wildcard, user_id)
        except UniqueViolation as error:
            # The (user_id, name) unique index is the authority on uniqueness, so a concurrent create loses here
            # rather than in a check-then-insert race. (An owner that does not exist is a ForeignKeyViolation.)
            raise _name_taken(wildcard.name) from error
        # The record is what was stored, so it is not read back.
        return WildcardRecordDTO(**wildcard.model_dump(), id=wildcard_id, user_id=user_id)

    def update(self, wildcard_id: str, changes: WildcardChanges) -> WildcardRecordDTO:
        def apply(q: Queries) -> Optional[WildcardRecordDTO]:
            try:
                q.wildcards.update(wildcard_id, changes)
            except UniqueViolation as error:
                raise _name_taken(changes.name) from error
            return q.wildcards.get(wildcard_id)

        return _found(self._queries.run(apply), wildcard_id)

    def delete(self, wildcard_id: str) -> None:
        self._queries.wildcards.delete(wildcard_id)

    def get_many(self, user_id: str) -> list[WildcardRecordDTO]:
        return self._queries.wildcards.owned_by(user_id)
