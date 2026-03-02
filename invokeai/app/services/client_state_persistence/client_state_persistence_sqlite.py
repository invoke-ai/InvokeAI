from invokeai.app.services.client_state_persistence.client_state_persistence_base import ClientStatePersistenceABC
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class ClientStatePersistenceSqlite(ClientStatePersistenceABC):
    """
    SQLite implementation for client state persistence.
    This class stores client state data per user to prevent data leakage between users.
    """

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def set_by_key(self, user_id: str, key: str, value: str) -> str:
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO client_state (user_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE
                  SET value = excluded.value;
                """,
                (user_id, key, value),
            )

        return value

    def get_by_key(self, user_id: str, key: str) -> str | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT value FROM client_state
                WHERE user_id = ? AND key = ?
                """,
                (user_id, key),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return row[0]

    def delete(self, user_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                DELETE FROM client_state
                WHERE user_id = ?
                """,
                (user_id,),
            )
