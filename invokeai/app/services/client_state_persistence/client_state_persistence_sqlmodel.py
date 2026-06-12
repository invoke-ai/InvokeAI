from invokeai.app.services.client_state_persistence.client_state_persistence_base import ClientStatePersistenceABC
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class ClientStatePersistenceSqlModel(ClientStatePersistenceABC):
    """SQLModel implementation for client state persistence."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def set_by_key(self, user_id: str, key: str, value: str) -> str:
        self._q.client_state_set_by_key(user_id, key, value)
        return value

    def get_by_key(self, user_id: str, key: str) -> str | None:
        return self._q.client_state_get_by_key(user_id, key)

    def get_keys_by_prefix(self, user_id: str, prefix: str) -> list[str]:
        return self._q.client_state_get_keys_by_prefix(user_id, prefix)

    def delete_by_key(self, user_id: str, key: str) -> None:
        self._q.client_state_delete_by_key(user_id, key)

    def delete(self, user_id: str) -> None:
        self._q.client_state_delete_all(user_id)
