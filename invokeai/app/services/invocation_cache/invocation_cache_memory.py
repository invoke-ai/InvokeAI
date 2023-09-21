from queue import Queue
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.app.services.invoker import Invoker


class MemoryInvocationCache(InvocationCacheBase):
    __cache: dict[Union[int, str], tuple[BaseInvocationOutput, str]]
    __max_cache_size: int
    __disabled: bool
    __hits: int
    __misses: int
    __cache_ids: Queue
    __invoker: Invoker

    def __init__(self, max_cache_size: int = 0) -> None:
        self.__cache = dict()
        self.__max_cache_size = max_cache_size
        self.__disabled = False
        self.__hits = 0
        self.__misses = 0
        self.__cache_ids = Queue()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        if self.__max_cache_size == 0:
            return
        self.__invoker.services.images.on_deleted(self._delete_by_match)
        self.__invoker.services.latents.on_deleted(self._delete_by_match)

    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        if self.__max_cache_size == 0 or self.__disabled:
            return

        item = self.__cache.get(key, None)
        if item is not None:
            self.__hits += 1
            return item[0]
        self.__misses += 1

    def save(self, key: Union[int, str], invocation_output: BaseInvocationOutput) -> None:
        if self.__max_cache_size == 0 or self.__disabled:
            return

        if key not in self.__cache:
            self.__cache[key] = (invocation_output, invocation_output.json())
            self.__cache_ids.put(key)
            if self.__cache_ids.qsize() > self.__max_cache_size:
                try:
                    self.__cache.pop(self.__cache_ids.get())
                except KeyError:
                    # this means the cache_ids are somehow out of sync w/ the cache
                    pass

    def delete(self, key: Union[int, str]) -> None:
        if self.__max_cache_size == 0 or self.__disabled:
            return

        if key in self.__cache:
            del self.__cache[key]

    def clear(self, *args, **kwargs) -> None:
        if self.__max_cache_size == 0 or self.__disabled:
            return

        self.__cache.clear()
        self.__cache_ids = Queue()
        self.__misses = 0
        self.__hits = 0

    def create_key(self, invocation: BaseInvocation) -> int:
        return hash(invocation.json(exclude={"id"}))

    def disable(self) -> None:
        self.__disabled = True

    def enable(self) -> None:
        self.__disabled = False

    def get_status(self) -> InvocationCacheStatus:
        return InvocationCacheStatus(
            hits=self.__hits,
            misses=self.__misses,
            enabled=not self.__disabled,
            size=len(self.__cache),
            max_size=self.__max_cache_size,
        )

    def _delete_by_match(self, to_match: str) -> None:
        if self.__max_cache_size == 0 or self.__disabled:
            return

        keys_to_delete = set()
        for key, value_tuple in self.__cache.items():
            if to_match in value_tuple[1]:
                keys_to_delete.add(key)

        if not keys_to_delete:
            return

        for key in keys_to_delete:
            self.delete(key)

        self.__invoker.services.logger.debug(f"Deleted {len(keys_to_delete)} cached invocation outputs for {to_match}")
