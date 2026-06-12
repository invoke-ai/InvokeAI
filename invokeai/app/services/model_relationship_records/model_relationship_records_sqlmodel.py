from invokeai.app.services.model_relationship_records.model_relationship_records_base import (
    ModelRelationshipRecordStorageBase,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqlModelModelRelationshipRecordStorage(ModelRelationshipRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        if model_key_1 == model_key_2:
            raise ValueError("Cannot relate a model to itself.")
        self._q.model_relationships_add(model_key_1, model_key_2)

    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        self._q.model_relationships_remove(model_key_1, model_key_2)

    def get_related_model_keys(self, model_key: str) -> list[str]:
        return self._q.model_relationships_get_related(model_key)

    def get_related_model_keys_batch(self, model_keys: list[str]) -> list[str]:
        return self._q.model_relationships_get_related_batch(model_keys)
