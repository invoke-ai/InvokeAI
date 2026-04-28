from typing import Optional

from invokeai.app.services.shared.sqlite.models import AppSettingTable
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class AppSettingsServiceSqlModel:
    """SQLModel implementation for application-level settings."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def get(self, key: str) -> Optional[str]:
        try:
            with self._db.get_readonly_session() as session:
                row = session.get(AppSettingTable, key)
                return row.value if row else None
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(AppSettingTable, key)
            if existing is not None:
                existing.value = value
                session.add(existing)
            else:
                session.add(AppSettingTable(key=key, value=value))

    def get_jwt_secret(self) -> str:
        secret = self.get("jwt_secret")
        if secret is None:
            raise RuntimeError(
                "JWT secret not found in database. This should have been created during database migration. "
                "Please ensure database migrations have been run successfully."
            )
        return secret
