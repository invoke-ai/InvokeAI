"""Tests for AppSettingsServiceSqlModel."""

import pytest

from invokeai.app.services.app_settings.app_settings_sqlmodel import AppSettingsServiceSqlModel
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


@pytest.fixture
def app_settings(db: SqliteDatabase) -> AppSettingsServiceSqlModel:
    return AppSettingsServiceSqlModel(db=db)


def test_get_nonexistent_key(app_settings: AppSettingsServiceSqlModel):
    assert app_settings.get("nonexistent") is None


def test_set_and_get(app_settings: AppSettingsServiceSqlModel):
    app_settings.set("test_key", "test_value")
    assert app_settings.get("test_key") == "test_value"


def test_set_overwrites_existing(app_settings: AppSettingsServiceSqlModel):
    app_settings.set("key", "value1")
    app_settings.set("key", "value2")
    assert app_settings.get("key") == "value2"


def test_get_jwt_secret(app_settings: AppSettingsServiceSqlModel):
    # jwt_secret is created by migration 27
    secret = app_settings.get_jwt_secret()
    assert secret is not None
    assert len(secret) > 0


def test_multiple_keys(app_settings: AppSettingsServiceSqlModel):
    app_settings.set("key1", "val1")
    app_settings.set("key2", "val2")
    assert app_settings.get("key1") == "val1"
    assert app_settings.get("key2") == "val2"
