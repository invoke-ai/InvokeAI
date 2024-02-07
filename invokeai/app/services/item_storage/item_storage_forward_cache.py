from queue import Queue
from typing import Optional, TypeVar

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.item_storage.item_storage_base import ItemStorageABC

T = TypeVar("T")


class ItemStorageForwardCache(ItemStorageABC[T]):
    """Provides a simple forward cache for an underlying storage. The cache is LRU and has a maximum size."""

    def __init__(self, underlying_storage: ItemStorageABC[T], max_cache_size: int = 20):
        super().__init__()
        self._underlying_storage = underlying_storage
        self._cache: dict[str, T] = {}
        self._cache_ids = Queue[str]()
        self._max_cache_size = max_cache_size

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        start_op = getattr(self._underlying_storage, "start", None)
        if callable(start_op):
            start_op(invoker)

    def stop(self, invoker: Invoker) -> None:
        self._invoker = invoker
        stop_op = getattr(self._underlying_storage, "stop", None)
        if callable(stop_op):
            stop_op(invoker)

    def get(self, item_id: str) -> T:
        cache_item = self._get_cache(item_id)
        if cache_item is not None:
            return cache_item

        latent = self._underlying_storage.get(item_id)
        self._set_cache(item_id, latent)
        return latent

    def set(self, item: T) -> str:
        item_id = self._underlying_storage.set(item)
        self._set_cache(item_id, item)
        self._on_changed(item)
        return item_id

    def delete(self, item_id: str) -> None:
        self._underlying_storage.delete(item_id)
        if item_id in self._cache:
            del self._cache[item_id]
        self._on_deleted(item_id)

    def _get_cache(self, item_id: str) -> Optional[T]:
        return None if item_id not in self._cache else self._cache[item_id]

    def _set_cache(self, item_id: str, data: T):
        if item_id not in self._cache:
            self._cache[item_id] = data
            self._cache_ids.put(item_id)
            if self._cache_ids.qsize() > self._max_cache_size:
                self._cache.pop(self._cache_ids.get())
