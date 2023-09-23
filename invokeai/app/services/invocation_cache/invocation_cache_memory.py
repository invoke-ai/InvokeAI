from queue import Queue
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.app.services.invoker import Invoker


class MemoryInvocationCache(InvocationCacheBase):
    _cache: dict[Union[int, str], tuple[BaseInvocationOutput, str]]
    _max_cache_size: int
    _disabled: bool
    _hits: int
    _misses: int
    _cache_ids: Queue
    _invoker: Invoker

    def __init__(self, max_cache_size: int = 0) -> None:
        self._cache = dict()
        self._max_cache_size = max_cache_size
        self._disabled = False
        self._hits = 0
        self._misses = 0
        self._cache_ids = Queue()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        if self._max_cache_size == 0:
            return
        self._invoker.services.images.on_deleted(self._delete_by_match)
        self._invoker.services.latents.on_deleted(self._delete_by_match)

    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        if self._max_cache_size == 0 or self._disabled:
            return

        item = self._cache.get(key, None)
        if item is not None:
            self._hits += 1
            return item[0]
        self._misses += 1

    def save(self, key: Union[int, str], invocation_output: BaseInvocationOutput) -> None:
        if self._max_cache_size == 0 or self._disabled:
            return

        if key not in self._cache:
            self._cache[key] = (invocation_output, invocation_output.json())
            self._cache_ids.put(key)
            if self._cache_ids.qsize() > self._max_cache_size:
                try:
                    self._cache.pop(self._cache_ids.get())
                except KeyError:
                    # this means the cache_ids are somehow out of sync w/ the cache
                    pass

    def delete(self, key: Union[int, str]) -> None:
        if self._max_cache_size == 0:
            return

        if key in self._cache:
            del self._cache[key]

    def clear(self, *args, **kwargs) -> None:
        if self._max_cache_size == 0:
            return

        self._cache.clear()
        self._cache_ids = Queue()
        self._misses = 0
        self._hits = 0

    def create_key(self, invocation: BaseInvocation) -> int:
        return hash(invocation.json(exclude={"id"}))

    def disable(self) -> None:
        if self._max_cache_size == 0:
            return
        self._disabled = True

    def enable(self) -> None:
        if self._max_cache_size == 0:
            return
        self._disabled = False

    def get_status(self) -> InvocationCacheStatus:
        return InvocationCacheStatus(
            hits=self._hits,
            misses=self._misses,
            enabled=not self._disabled and self._max_cache_size > 0,
            size=len(self._cache),
            max_size=self._max_cache_size,
        )

    def _delete_by_match(self, to_match: str) -> None:
        if self._max_cache_size == 0:
            return

        keys_to_delete = set()
        for key, value_tuple in self._cache.items():
            if to_match in value_tuple[1]:
                keys_to_delete.add(key)

        if not keys_to_delete:
            return

        for key in keys_to_delete:
            self.delete(key)

        self._invoker.services.logger.debug(f"Deleted {len(keys_to_delete)} cached invocation outputs for {to_match}")
