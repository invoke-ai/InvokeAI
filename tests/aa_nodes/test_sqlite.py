import pytest
from pydantic import BaseModel, Field

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.item_storage.item_storage_sqlite import SqliteItemStorage
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger


class TestModel(BaseModel):
    id: str = Field(description="ID")
    name: str = Field(description="Name")
    __test__ = False  # not a pytest test case


@pytest.fixture
def db() -> SqliteItemStorage[TestModel]:
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger()
    db_path = None if config.use_memory_db else config.db_path
    db = SqliteDatabase(db_path=db_path, logger=logger, verbose=config.log_sql)
    sqlite_item_storage = SqliteItemStorage[TestModel](db=db, table_name="test", id_field="id")
    return sqlite_item_storage


def test_sqlite_service_can_create_and_get(db: SqliteItemStorage[TestModel]):
    db.set(TestModel(id="1", name="Test"))
    assert db.get("1") == TestModel(id="1", name="Test")


def test_sqlite_service_can_delete(db: SqliteItemStorage[TestModel]):
    db.set(TestModel(id="1", name="Test"))
    db.delete("1")
    assert db.get("1") is None


def test_sqlite_service_calls_set_callback(db: SqliteItemStorage[TestModel]):
    called = False

    def on_changed(item: TestModel):
        nonlocal called
        called = True

    db.on_changed(on_changed)
    db.set(TestModel(id="1", name="Test"))
    assert called


def test_sqlite_service_calls_delete_callback(db: SqliteItemStorage[TestModel]):
    called = False

    def on_deleted(item_id: str):
        nonlocal called
        called = True

    db.on_deleted(on_deleted)
    db.set(TestModel(id="1", name="Test"))
    db.delete("1")
    assert called
