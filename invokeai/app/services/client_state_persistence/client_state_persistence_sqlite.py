import json

from pydantic import JsonValue

from invokeai.app.services.client_state_persistence.client_state_persistence_base import ClientStatePersistenceABC
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class ClientStatePersistenceSqlite(ClientStatePersistenceABC):
    """
    Base class for client persistence implementations.
    This class defines the interface for persisting client data.
    """

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._default_row_id = 1

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def set_by_key(self, key: str, value: JsonValue) -> None:
        state = self.get() or {}
        state.update({key: value})

        with self._db.transaction() as cursor:
            cursor.execute(
                f"""
                INSERT INTO client_state (id, data)
                VALUES ({self._default_row_id}, ?)
                ON CONFLICT(id) DO UPDATE
                  SET data = excluded.data;
                """,
                (json.dumps(state),),
            )

    def get(self) -> dict[str, JsonValue] | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""
                SELECT data FROM client_state
                WHERE id = {self._default_row_id}
                """
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def get_by_key(self, key: str) -> JsonValue | None:
        state = self.get()
        if state is None:
            return None
        return state.get(key, None)

    def delete(self) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""
                DELETE FROM client_state
                WHERE id = {self._default_row_id}
                """
            )
