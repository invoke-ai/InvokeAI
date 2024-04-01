from queue import Queue
from typing import TYPE_CHECKING, Optional, TypeVar
import threading

from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase

T = TypeVar("T")

if TYPE_CHECKING:
    from invokeai.app.services.invoker import Invoker


class ObjectSerializerForwardCache(ObjectSerializerBase[T]):
    """
    Provides a LRU cache for an instance of `ObjectSerializerBase`.
    Saving an object to the cache always writes through to the underlying storage.
    """

    def __init__(self, underlying_storage: ObjectSerializerBase[T], max_cache_size: int = 20):
        super().__init__()
        self._underlying_storage = underlying_storage
        self._cache: dict[int, dict[str, T]] = {}
        self._cache_ids: dict[int, Queue[str]] = {}
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

        obj = self._underlying_storage.load(name)
        self._set_cache(name, obj)
        return obj

    def save(self, obj: T) -> str:
        name = self._underlying_storage.save(obj)
        self._set_cache(name, obj)
        return name

    def delete(self, name: str) -> None:
        self._underlying_storage.delete(name)
        if name in self._cache:
            del self._cache[name]
        self._on_deleted(name)

    def _get_tid_cache(self) -> dict[str, T]:
        tid = threading.current_thread().ident
        if tid not in self._cache:
            self._cache[tid] = {}
        return self._cache[tid]

    def _get_tid_cache_ids(self) -> Queue[str]:
        tid = threading.current_thread().ident
        if tid not in self._cache_ids:
            self._cache_ids[tid] = Queue[str]()
        return self._cache_ids[tid]

    def _get_cache(self, name: str) -> Optional[T]:
        cache = self._get_tid_cache()
        return None if name not in cache else cache[name]

    def _set_cache(self, name: str, data: T):
        cache = self._get_tid_cache()
        if name not in cache:
            cache[name] = data
            cache_ids = self._get_tid_cache_ids()
            cache_ids.put(name)
            if cache_ids.qsize() > self._max_cache_size:
                cache.pop(cache_ids.get())
