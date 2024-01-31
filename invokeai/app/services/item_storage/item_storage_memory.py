from contextlib import suppress
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

from invokeai.app.services.item_storage.item_storage_base import ItemStorageABC

T = TypeVar("T", bound=BaseModel)


class ItemStorageMemory(ItemStorageABC, Generic[T]):
    """
    Provides a simple in-memory storage for items, with a maximum number of items to store.
    An item is deleted when the maximum number of items is reached and a new item is added.
    There is no guarantee about which item will be deleted.
    """

    def __init__(self, id_field: str = "id", max_items: int = 10) -> None:
        super().__init__()
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if not id_field:
            raise ValueError("id_field must not be empty")
        self._id_field = id_field
        self._items: dict[str, T] = {}
        self._item_ids: set[str] = set()
        self._max_items = max_items

    def get(self, item_id: str) -> Optional[T]:
        return self._items.get(item_id)

    def set(self, item: T) -> None:
        item_id = getattr(item, self._id_field)
        assert isinstance(item_id, str)
        if item_id in self._items or len(self._items) < self._max_items:
            # If the item is already stored, or we have room for more items, we can just add it.
            self._items[item_id] = item
            self._item_ids.add(item_id)
        else:
            # Otherwise, we need to make room for it first.
            self._items.pop(self._item_ids.pop())
            self._items[item_id] = item
            self._item_ids.add(item_id)
        self._on_changed(item)

    def delete(self, item_id: str) -> None:
        # Both of these are no-ops if the item doesn't exist.
        with suppress(KeyError):
            del self._items[item_id]
            self._item_ids.remove(item_id)
            self._on_deleted(item_id)
