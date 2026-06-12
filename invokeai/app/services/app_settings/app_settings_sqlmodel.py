from typing import Optional

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class AppSettingsServiceSqlModel:
    """SQLModel implementation for application-level settings."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db
        self._q = db.queries

    def get(self, key: str) -> Optional[str]:
        try:
            return self._q.app_settings_get(key)
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        self._q.app_settings_set(key, value)

    def get_jwt_secret(self) -> str:
        secret = self.get("jwt_secret")
        if secret is None:
            raise RuntimeError(
                "JWT secret not found in database. This should have been created during database migration. "
                "Please ensure database migrations have been run successfully."
            )
        return secret
