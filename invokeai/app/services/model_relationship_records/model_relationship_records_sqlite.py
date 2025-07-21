from invokeai.app.services.model_relationship_records.model_relationship_records_base import (
    ModelRelationshipRecordStorageBase,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteModelRelationshipRecordStorage(ModelRelationshipRecordStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        with self._db.transaction() as cursor:
            if model_key_1 == model_key_2:
                raise ValueError("Cannot relate a model to itself.")
            a, b = sorted([model_key_1, model_key_2])
            cursor.execute(
                "INSERT OR IGNORE INTO model_relationships (model_key_1, model_key_2) VALUES (?, ?)",
                (a, b),
            )

    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        with self._db.transaction() as cursor:
            a, b = sorted([model_key_1, model_key_2])
            cursor.execute(
                "DELETE FROM model_relationships WHERE model_key_1 = ? AND model_key_2 = ?",
                (a, b),
            )

    def get_related_model_keys(self, model_key: str) -> list[str]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT model_key_2 FROM model_relationships WHERE model_key_1 = ?
                UNION
                SELECT model_key_1 FROM model_relationships WHERE model_key_2 = ?
                """,
                (model_key, model_key),
            )
            result = [row[0] for row in cursor.fetchall()]
        return result

    def get_related_model_keys_batch(self, model_keys: list[str]) -> list[str]:
        with self._db.transaction() as cursor:
            key_list = ",".join("?" for _ in model_keys)
            cursor.execute(
                f"""
                SELECT model_key_2 FROM model_relationships WHERE model_key_1 IN ({key_list})
                UNION
                SELECT model_key_1 FROM model_relationships WHERE model_key_2 IN ({key_list})
                """,
                model_keys + model_keys,
            )
            result = [row[0] for row in cursor.fetchall()]
        return result
