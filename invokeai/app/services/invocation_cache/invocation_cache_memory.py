from dataclasses import dataclass, field
from time import time
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus, ThreadLock
from invokeai.app.services.invoker import Invoker

thread_lock = ThreadLock()


@dataclass(order=True)
class CachedItem:
    priority: float
    invocation_output: BaseInvocationOutput = field(compare=False)
    invocation_output_json: str = field(compare=False)


class MemoryInvocationCache(InvocationCacheBase):
    _cache: dict[Union[int, str], CachedItem]
    _max_cache_size: int
    _disabled: bool
    _hits: int
    _misses: int
    _invoker: Invoker

    def __init__(self, max_cache_size: int = 0) -> None:
        self._cache = dict()
        self._max_cache_size = max_cache_size
        self._disabled = False
        self._hits = 0
        self._misses = 0

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        if self._max_cache_size == 0:
            return
        self._invoker.services.images.on_deleted(self._delete_by_match)
        self._invoker.services.latents.on_deleted(self._delete_by_match)

    @thread_lock.write  # because of priority update
    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        if self._max_cache_size == 0 or self._disabled:
            return
        item = self._cache.get(key, None)
        if item is not None:
            self._hits += 1
            item.priority = time()
            return item.invocation_output
        self._misses += 1

    @thread_lock.write
    def save(self, key: Union[int, str], invocation_output: BaseInvocationOutput) -> None:
        if self._max_cache_size == 0 or self._disabled or key in self._cache:
            return
        # If the cache is full, we need to remove the least used
        number_to_delete = len(self._cache) + 1 - self._max_cache_size
        self._delete_least_accessed(number_to_delete)
        self._cache[key] = CachedItem(time(), invocation_output, invocation_output.json())

    def _delete_oldest_access(self, number_to_delete: int) -> None:
        number_to_delete = min(number_to_delete, len(self._cache))
        if self._max_cache_size == 0 or number_to_delete < 1:
            return
        keys_by_access_time = sorted(self._cache.keys(), key=lambda x: self._cache[x].priority)
        for index in range(number_to_delete):
            self._delete(keys_by_access_time[index])

    def _delete(self, key: Union[int, str]) -> None:
        if self._max_cache_size == 0:
            return
        if key in self._cache:
            del self._cache[key]

    @thread_lock.write
    def delete(self, key: Union[int, str]) -> None:
        return self._delete(key)

    @thread_lock.write
    def clear(self, *args, **kwargs) -> None:
        if self._max_cache_size == 0:
            return
        self._cache.clear()
        self._misses = 0
        self._hits = 0

    @staticmethod
    def create_key(invocation: BaseInvocation) -> int:
        return hash(invocation.json(exclude={"id"}))

    @thread_lock.write
    def disable(self) -> None:
        if self._max_cache_size == 0:
            return
        self._disabled = True

    @thread_lock.write
    def enable(self) -> None:
        if self._max_cache_size == 0:
            return
        self._disabled = False

    @thread_lock.read
    def get_status(self) -> InvocationCacheStatus:
        return InvocationCacheStatus(
            hits=self._hits,
            misses=self._misses,
            enabled=not self._disabled and self._max_cache_size > 0,
            size=len(self._cache),
            max_size=self._max_cache_size,
        )

    @thread_lock.write
    def _delete_by_match(self, to_match: str) -> None:
        if self._max_cache_size == 0:
            return
        keys_to_delete = set()
        for key, cached_item in self._cache.items():
            if to_match in cached_item.invocation_output_json:
                keys_to_delete.add(key)
        if not keys_to_delete:
            return
        for key in keys_to_delete:
            self._delete(key)
        self._invoker.services.logger.debug(f"Deleted {len(keys_to_delete)} cached invocation outputs for {to_match}")
