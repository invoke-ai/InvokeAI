"""Server-wide settings kept in the database."""

from typing import Optional

from sqlalchemy import Connection, bindparam, select

from invokeai.app.services.shared.database.queries.base import QueryModule, read
from invokeai.app.services.shared.database.schema.app_settings import app_settings

_GET = select(app_settings.c.value).where(app_settings.c.key == bindparam("key"))


class AppSettingQueries(QueryModule):
    @read
    def get(self, conn: Connection, key: str) -> Optional[str]:
        return conn.execute(_GET, {"key": key}).scalar_one_or_none()
