from queue import Queue
from typing import TYPE_CHECKING, Optional, TypeVar

from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase

T = TypeVar("T")

if TYPE_CHECKING:
    from invokeai.app.services.invoker import Invoker


class ObjectSerializerForwardCache(ObjectSerializerBase[T]):
    """Provides a simple forward cache for an underlying storage. The cache is LRU and has a maximum size."""

    def __init__(self, underlying_storage: ObjectSerializerBase[T], max_cache_size: int = 20):
        super().__init__()
        self._underlying_storage = underlying_storage
        self._cache: dict[str, T] = {}
        self._cache_ids = Queue[str]()
        self._max_cache_size = max_cache_size

    def start(self, invoker: "Invoker") -> None:
        self._invoker = invoker
        start_op = getattr(self._underlying_storage, "start", None)
        if callable(start_op):
            start_op(invoker)

    def stop(self, invoker: "Invoker") -> None:
        self._invoker = invoker
        stop_op = getattr(self._underlying_storage, "stop", None)
        if callable(stop_op):
            stop_op(invoker)

    def load(self, name: str) -> T:
        cache_item = self._get_cache(name)
        if cache_item is not None:
            return cache_item

        latent = self._underlying_storage.load(name)
        self._set_cache(name, latent)
        return latent

    def save(self, obj: T) -> str:
        name = self._underlying_storage.save(obj)
        self._set_cache(name, obj)
        return name

    def delete(self, name: str) -> None:
        self._underlying_storage.delete(name)
        if name in self._cache:
            del self._cache[name]
        self._on_deleted(name)

    def _get_cache(self, name: str) -> Optional[T]:
        return None if name not in self._cache else self._cache[name]

    def _set_cache(self, name: str, data: T):
        if name not in self._cache:
            self._cache[name] = data
            self._cache_ids.put(name)
            if self._cache_ids.qsize() > self._max_cache_size:
                self._cache.pop(self._cache_ids.get())
