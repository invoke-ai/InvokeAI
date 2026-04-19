from sqlmodel import col, select

from invokeai.app.services.client_state_persistence.client_state_persistence_base import ClientStatePersistenceABC
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.models import ClientStateTable
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class ClientStatePersistenceSqlModel(ClientStatePersistenceABC):
    """SQLModel implementation for client state persistence."""

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def set_by_key(self, user_id: str, key: str, value: str) -> str:
        with self._db.get_session() as session:
            existing = session.get(ClientStateTable, (user_id, key))
            if existing is not None:
                existing.value = value
                session.add(existing)
            else:
                session.add(ClientStateTable(user_id=user_id, key=key, value=value))
        return value

    def get_by_key(self, user_id: str, key: str) -> str | None:
        with self._db.get_readonly_session() as session:
            row = session.get(ClientStateTable, (user_id, key))
            if row is None:
                return None
            return row.value

    def delete(self, user_id: str) -> None:
        with self._db.get_session() as session:
            stmt = select(ClientStateTable).where(col(ClientStateTable.user_id) == user_id)
            rows = session.exec(stmt).all()
            for row in rows:
                session.delete(row)
