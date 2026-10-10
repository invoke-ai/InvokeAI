from invokeai.app.services.model_relationship_records.model_relationship_records_base import (
    ModelRelationshipRecordStorageBase,
)
from invokeai.app.services.shared.database.database import Database


class ModelRelationshipRecordStorage(ModelRelationshipRecordStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        if model_key_1 == model_key_2:
            raise ValueError("Cannot relate a model to itself.")
        self._queries.model_relationships.add(model_key_1, model_key_2)

    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        self._queries.model_relationships.remove(model_key_1, model_key_2)

    def get_related_model_keys(self, model_key: str) -> list[str]:
        return self._queries.model_relationships.related(model_key)
