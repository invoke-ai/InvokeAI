from collections import OrderedDict
from contextlib import suppress
from typing import Generic, TypeVar

from pydantic import BaseModel

from invokeai.app.services.item_storage.item_storage_base import ItemStorageABC
from invokeai.app.services.item_storage.item_storage_common import ItemNotFoundError

T = TypeVar("T", bound=BaseModel)


class ItemStorageMemory(ItemStorageABC[T], Generic[T]):
    """
    Provides a simple in-memory storage for items, with a maximum number of items to store.
    The storage uses the LRU strategy to evict items from storage when the max has been reached.
    """

    def __init__(self, id_field: str = "id", max_items: int = 10) -> None:
        super().__init__()
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        if not id_field:
            raise ValueError("id_field must not be empty")
        self._id_field = id_field
        self._items: OrderedDict[str, T] = OrderedDict()
        self._max_items = max_items

    def get(self, item_id: str) -> T:
        # If the item exists, move it to the end of the OrderedDict.
        item = self._items.pop(item_id, None)
        if item is None:
            raise ItemNotFoundError(item_id)
        self._items[item_id] = item
        return item

    def set(self, item: T) -> None:
        item_id = getattr(item, self._id_field)
        if item_id in self._items:
            # If item already exists, remove it and add it to the end
            self._items.pop(item_id)
        elif len(self._items) >= self._max_items:
            # If cache is full, evict the least recently used item
            self._items.popitem(last=False)
        self._items[item_id] = item
        self._on_changed(item)

    def delete(self, item_id: str) -> None:
        # This is a no-op if the item doesn't exist.
        with suppress(KeyError):
            del self._items[item_id]
            self._on_deleted(item_id)
