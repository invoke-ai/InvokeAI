import json
import sqlite3

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.wildcard_records.wildcard_records_base import WildcardRecordsStorageBase
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardRecordDTO,
    WildcardWithoutId,
)
from invokeai.app.util.misc import uuid_string


def _is_name_conflict(error: sqlite3.IntegrityError) -> bool:
    """Whether an integrity failure is the duplicate name and not, say, an unknown owner.

    The table also carries a users foreign key, and reporting a missing owner as
    "that name is taken" would send the caller chasing the wrong thing.
    """
    return "UNIQUE constraint failed" in str(error)


class SqliteWildcardRecordsStorage(WildcardRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def get(self, wildcard_id: str) -> WildcardRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT id, name, values_json AS "values", user_id
                FROM wildcards
                WHERE id = ?;
                """,
                (wildcard_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise WildcardNotFoundError(f"Wildcard with id {wildcard_id} not found")
        return WildcardRecordDTO.from_dict(dict(row))

    def create(self, wildcard: WildcardWithoutId, user_id: str) -> WildcardRecordDTO:
        wildcard_id = uuid_string()
        try:
            with self._db.transaction() as cursor:
                cursor.execute(
                    """--sql
                    INSERT INTO wildcards (id, name, values_json, user_id)
                    VALUES (?, ?, ?, ?);
                    """,
                    (wildcard_id, wildcard.name, json.dumps(wildcard.values), user_id),
                )
        except sqlite3.IntegrityError as e:
            # The (user_id, name) unique index is the authority on uniqueness, so a concurrent
            # create loses here rather than in a check-then-insert race.
            if not _is_name_conflict(e):
                raise
            raise WildcardNameConflictError(f"A wildcard named '{wildcard.name}' already exists") from e
        return self.get(wildcard_id)

    def update(self, wildcard_id: str, changes: WildcardChanges) -> WildcardRecordDTO:
        try:
            with self._db.transaction() as cursor:
                if changes.name is not None:
                    cursor.execute(
                        """--sql
                        UPDATE wildcards SET name = ? WHERE id = ?;
                        """,
                        (changes.name, wildcard_id),
                    )
                if changes.values is not None:
                    cursor.execute(
                        """--sql
                        UPDATE wildcards SET values_json = ? WHERE id = ?;
                        """,
                        (json.dumps(changes.values), wildcard_id),
                    )
        except sqlite3.IntegrityError as e:
            if not _is_name_conflict(e):
                raise
            raise WildcardNameConflictError(f"A wildcard named '{changes.name}' already exists") from e
        return self.get(wildcard_id)

    def delete(self, wildcard_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM wildcards WHERE id = ?;
                """,
                (wildcard_id,),
            )

    def get_many(self, user_id: str) -> list[WildcardRecordDTO]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT id, name, values_json AS "values", user_id
                FROM wildcards
                WHERE user_id = ?
                ORDER BY name ASC;
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
        return [WildcardRecordDTO.from_dict(dict(row)) for row in rows]
