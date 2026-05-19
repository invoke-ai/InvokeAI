from sqlmodel import col, select

from invokeai.app.services.model_relationship_records.model_relationship_records_base import (
    ModelRelationshipRecordStorageBase,
)
from invokeai.app.services.shared.sqlite.models import ModelRelationshipTable
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqlModelModelRelationshipRecordStorage(ModelRelationshipRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        if model_key_1 == model_key_2:
            raise ValueError("Cannot relate a model to itself.")
        a, b = sorted([model_key_1, model_key_2])
        with self._db.get_session() as session:
            existing = session.get(ModelRelationshipTable, (a, b))
            if existing is None:
                session.add(ModelRelationshipTable(model_key_1=a, model_key_2=b))

    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        a, b = sorted([model_key_1, model_key_2])
        with self._db.get_session() as session:
            existing = session.get(ModelRelationshipTable, (a, b))
            if existing is not None:
                session.delete(existing)

    def get_related_model_keys(self, model_key: str) -> list[str]:
        with self._db.get_readonly_session() as session:
            # Get keys where model_key appears in either column
            stmt1 = select(ModelRelationshipTable.model_key_2).where(
                col(ModelRelationshipTable.model_key_1) == model_key
            )
            stmt2 = select(ModelRelationshipTable.model_key_1).where(
                col(ModelRelationshipTable.model_key_2) == model_key
            )
            results1 = session.exec(stmt1).all()
            results2 = session.exec(stmt2).all()
        return list(set(results1 + results2))

    def get_related_model_keys_batch(self, model_keys: list[str]) -> list[str]:
        with self._db.get_readonly_session() as session:
            stmt1 = select(ModelRelationshipTable.model_key_2).where(
                col(ModelRelationshipTable.model_key_1).in_(model_keys)
            )
            stmt2 = select(ModelRelationshipTable.model_key_1).where(
                col(ModelRelationshipTable.model_key_2).in_(model_keys)
            )
            results1 = session.exec(stmt1).all()
            results2 = session.exec(stmt2).all()
        return list(set(results1 + results2))
