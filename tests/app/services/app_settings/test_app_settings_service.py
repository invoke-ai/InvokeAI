import pytest
from sqlalchemy import delete

from invokeai.app.services.app_settings import AppSettingsService
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.app_settings import app_settings


def test_the_jwt_secret_a_new_database_starts_with_is_read(database: Database) -> None:
    assert len(AppSettingsService(database).get_jwt_secret()) == 64


def test_a_database_without_a_jwt_secret_is_reported(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(delete(app_settings).where(app_settings.c.key == "jwt_secret"))

    with pytest.raises(RuntimeError, match="JWT secret not found"):
        AppSettingsService(database).get_jwt_secret()
