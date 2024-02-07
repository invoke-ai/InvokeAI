from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.app.services.invoker import Invoker


@dataclass(order=True)
class CachedItem:
    invocation_output: BaseInvocationOutput = field(compare=False)
    invocation_output_json: str = field(compare=False)


class MemoryInvocationCache(InvocationCacheBase):
    _cache: OrderedDict[Union[int, str], CachedItem]
    _max_cache_size: int
    _disabled: bool
    _hits: int
    _misses: int
    _invoker: Invoker
    _lock: Lock

    def __init__(self, max_cache_size: int = 0) -> None:
        self._cache = OrderedDict()
        self._max_cache_size = max_cache_size
        self._disabled = False
        self._hits = 0
        self._misses = 0
        self._lock = Lock()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        if self._max_cache_size == 0:
            return
        self._invoker.services.images.on_deleted(self._delete_by_match)
        self._invoker.services.tensors.on_deleted(self._delete_by_match)
        self._invoker.services.conditioning.on_deleted(self._delete_by_match)

    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        with self._lock:
            if self._max_cache_size == 0 or self._disabled:
                return None
            item = self._cache.get(key, None)
            if item is not None:
                self._hits += 1
                self._cache.move_to_end(key)
                return item.invocation_output
            self._misses += 1
            return None

    def save(self, key: Union[int, str], invocation_output: BaseInvocationOutput) -> None:
        with self._lock:
            if self._max_cache_size == 0 or self._disabled or key in self._cache:
                return
            # If the cache is full, we need to remove the least used
            number_to_delete = len(self._cache) + 1 - self._max_cache_size
            self._delete_oldest_access(number_to_delete)
            self._cache[key] = CachedItem(
                invocation_output,
                invocation_output.model_dump_json(
                    warnings=False, exclude_defaults=True, exclude_unset=True, include={"type"}
                ),
            )

    def _delete_oldest_access(self, number_to_delete: int) -> None:
        number_to_delete = min(number_to_delete, len(self._cache))
        for _ in range(number_to_delete):
            self._cache.popitem(last=False)

    def _delete(self, key: Union[int, str]) -> None:
        if self._max_cache_size == 0:
            return
        if key in self._cache:
            del self._cache[key]

    def delete(self, key: Union[int, str]) -> None:
        with self._lock:
            return self._delete(key)

    def clear(self, *args, **kwargs) -> None:
        with self._lock:
            if self._max_cache_size == 0:
                return
            self._cache.clear()
            self._misses = 0
            self._hits = 0

    @staticmethod
    def create_key(invocation: BaseInvocation) -> int:
        return hash(invocation.model_dump_json(exclude={"id"}, warnings=False))

    def disable(self) -> None:
        with self._lock:
            if self._max_cache_size == 0:
                return
            self._disabled = True

    def enable(self) -> None:
        with self._lock:
            if self._max_cache_size == 0:
                return
            self._disabled = False

    def get_status(self) -> InvocationCacheStatus:
        with self._lock:
            return InvocationCacheStatus(
                hits=self._hits,
                misses=self._misses,
                enabled=not self._disabled and self._max_cache_size > 0,
                size=len(self._cache),
                max_size=self._max_cache_size,
            )

    def _delete_by_match(self, to_match: str) -> None:
        with self._lock:
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
            self._invoker.services.logger.debug(
                f"Deleted {len(keys_to_delete)} cached invocation outputs for {to_match}"
            )
