import re

import pytest
from pydantic import BaseModel

from invokeai.app.services.item_storage.item_storage_common import ItemNotFoundError
from invokeai.app.services.item_storage.item_storage_memory import ItemStorageMemory


class MockItemModel(BaseModel):
    id: str
    value: int


@pytest.fixture
def item_storage_memory():
    return ItemStorageMemory[MockItemModel]()


def test_item_storage_memory_initializes():
    item_storage_memory = ItemStorageMemory[MockItemModel]()
    assert item_storage_memory._items == {}
    assert item_storage_memory._id_field == "id"
    assert item_storage_memory._max_items == 10

    item_storage_memory = ItemStorageMemory[MockItemModel](id_field="bananas", max_items=20)
    assert item_storage_memory._id_field == "bananas"
    assert item_storage_memory._max_items == 20

    with pytest.raises(ValueError, match=re.escape("max_items must be at least 1")):
        item_storage_memory = ItemStorageMemory[MockItemModel](max_items=0)
    with pytest.raises(ValueError, match=re.escape("id_field must not be empty")):
        item_storage_memory = ItemStorageMemory[MockItemModel](id_field="")


def test_item_storage_memory_sets(item_storage_memory: ItemStorageMemory[MockItemModel]):
    item_1 = MockItemModel(id="1", value=1)
    item_storage_memory.set(item_1)
    assert item_storage_memory._items == {"1": item_1}

    item_2 = MockItemModel(id="2", value=2)
    item_storage_memory.set(item_2)
    assert item_storage_memory._items == {"1": item_1, "2": item_2}

    # Updating value of existing item
    item_2_updated = MockItemModel(id="2", value=9001)
    item_storage_memory.set(item_2_updated)
    assert item_storage_memory._items == {"1": item_1, "2": item_2_updated}


def test_item_storage_memory_gets(item_storage_memory: ItemStorageMemory[MockItemModel]):
    item_1 = MockItemModel(id="1", value=1)
    item_storage_memory.set(item_1)
    item = item_storage_memory.get("1")
    assert item == item_1

    item_2 = MockItemModel(id="2", value=2)
    item_storage_memory.set(item_2)
    item = item_storage_memory.get("2")
    assert item == item_2

    with pytest.raises(ItemNotFoundError, match=re.escape("Item with id 3 not found")):
        item_storage_memory.get("3")


def test_item_storage_memory_deletes(item_storage_memory: ItemStorageMemory[MockItemModel]):
    item_1 = MockItemModel(id="1", value=1)
    item_2 = MockItemModel(id="2", value=2)
    item_storage_memory.set(item_1)
    item_storage_memory.set(item_2)

    item_storage_memory.delete("2")
    assert item_storage_memory._items == {"1": item_1}


def test_item_storage_memory_respects_max():
    item_storage_memory = ItemStorageMemory[MockItemModel](max_items=3)
    for i in range(10):
        item_storage_memory.set(MockItemModel(id=str(i), value=i))
    assert item_storage_memory._items == {
        "7": MockItemModel(id="7", value=7),
        "8": MockItemModel(id="8", value=8),
        "9": MockItemModel(id="9", value=9),
    }


def test_item_storage_memory_calls_set_callback(item_storage_memory: ItemStorageMemory[MockItemModel]):
    called_item = None
    item = MockItemModel(id="1", value=1)

    def on_changed(item: MockItemModel):
        nonlocal called_item
        called_item = item

    item_storage_memory.on_changed(on_changed)
    item_storage_memory.set(item)
    assert called_item == item


def test_item_storage_memory_calls_delete_callback(item_storage_memory: ItemStorageMemory[MockItemModel]):
    called_item_id = None
    item = MockItemModel(id="1", value=1)

    def on_deleted(item_id: str):
        nonlocal called_item_id
        called_item_id = item_id

    item_storage_memory.on_deleted(on_deleted)
    item_storage_memory.set(item)
    item_storage_memory.delete("1")
    assert called_item_id == "1"
