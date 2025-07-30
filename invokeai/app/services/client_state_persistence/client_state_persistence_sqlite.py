import json

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

    def _get(self) -> dict[str, str] | None:
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

    def set_by_key(self, queue_id: str, key: str, value: str) -> str:
        state = self._get() or {}
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

        return value

    def get_by_key(self, queue_id: str, key: str) -> str | None:
        state = self._get()
        if state is None:
            return None
        return state.get(key, None)

    def delete(self, queue_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""
                DELETE FROM client_state
                WHERE id = {self._default_row_id}
                """
            )
