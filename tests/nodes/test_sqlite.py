from invokeai.app.services.sqlite import SqliteItemStorage, sqlite_memory
from pydantic import BaseModel, Field


class TestModel(BaseModel):
    id: str = Field(description="ID")
    name: str = Field(description="Name")


def test_sqlite_service_can_create_and_get():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    assert db.get("1") == TestModel(id="1", name="Test")


def test_sqlite_service_can_list():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.set(TestModel(id="2", name="Test"))
    db.set(TestModel(id="3", name="Test"))
    results = db.list()
    assert results.page == 0
    assert results.pages == 1
    assert results.per_page == 10
    assert results.total == 3
    assert results.items == [
        TestModel(id="1", name="Test"),
        TestModel(id="2", name="Test"),
        TestModel(id="3", name="Test"),
    ]


def test_sqlite_service_can_delete():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.delete("1")
    assert db.get("1") is None


def test_sqlite_service_calls_set_callback():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    called = False

    def on_changed(item: TestModel):
        nonlocal called
        called = True

    db.on_changed(on_changed)
    db.set(TestModel(id="1", name="Test"))
    assert called


def test_sqlite_service_calls_delete_callback():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    called = False

    def on_deleted(item_id: str):
        nonlocal called
        called = True

    db.on_deleted(on_deleted)
    db.set(TestModel(id="1", name="Test"))
    db.delete("1")
    assert called


def test_sqlite_service_can_list_with_pagination():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.set(TestModel(id="2", name="Test"))
    db.set(TestModel(id="3", name="Test"))
    results = db.list(page=0, per_page=2)
    assert results.page == 0
    assert results.pages == 2
    assert results.per_page == 2
    assert results.total == 3
    assert results.items == [TestModel(id="1", name="Test"), TestModel(id="2", name="Test")]


def test_sqlite_service_can_list_with_pagination_and_offset():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.set(TestModel(id="2", name="Test"))
    db.set(TestModel(id="3", name="Test"))
    results = db.list(page=1, per_page=2)
    assert results.page == 1
    assert results.pages == 2
    assert results.per_page == 2
    assert results.total == 3
    assert results.items == [TestModel(id="3", name="Test")]


def test_sqlite_service_can_search():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.set(TestModel(id="2", name="Test"))
    db.set(TestModel(id="3", name="Test"))
    results = db.search(query="Test")
    assert results.page == 0
    assert results.pages == 1
    assert results.per_page == 10
    assert results.total == 3
    assert results.items == [
        TestModel(id="1", name="Test"),
        TestModel(id="2", name="Test"),
        TestModel(id="3", name="Test"),
    ]


def test_sqlite_service_can_search_with_pagination():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.set(TestModel(id="2", name="Test"))
    db.set(TestModel(id="3", name="Test"))
    results = db.search(query="Test", page=0, per_page=2)
    assert results.page == 0
    assert results.pages == 2
    assert results.per_page == 2
    assert results.total == 3
    assert results.items == [TestModel(id="1", name="Test"), TestModel(id="2", name="Test")]


def test_sqlite_service_can_search_with_pagination_and_offset():
    db = SqliteItemStorage[TestModel](sqlite_memory, "test", "id")
    db.set(TestModel(id="1", name="Test"))
    db.set(TestModel(id="2", name="Test"))
    db.set(TestModel(id="3", name="Test"))
    results = db.search(query="Test", page=1, per_page=2)
    assert results.page == 1
    assert results.pages == 2
    assert results.per_page == 2
    assert results.total == 3
    assert results.items == [TestModel(id="3", name="Test")]
