from queue import Queue
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase


class MemoryInvocationCache(InvocationCacheBase):
    __cache: dict[Union[int, str], BaseInvocationOutput]
    __max_cache_size: int
    __cache_ids: Queue

    def __init__(self, max_cache_size: int = 512) -> None:
        self.__cache = dict()
        self.__max_cache_size = max_cache_size
        self.__cache_ids = Queue()

    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        if self.__max_cache_size == 0:
            return None

        return self.__cache.get(key, None)

    def save(self, key: Union[int, str], value: BaseInvocationOutput) -> None:
        if self.__max_cache_size == 0:
            return None

        if key not in self.__cache:
            self.__cache[key] = value
            self.__cache_ids.put(key)
            if self.__cache_ids.qsize() > self.__max_cache_size:
                self.__cache.pop(self.__cache_ids.get())

    def delete(self, key: Union[int, str]) -> None:
        if self.__max_cache_size == 0:
            return None

        if key in self.__cache:
            del self.__cache[key]

    @classmethod
    def create_key(cls, value: BaseInvocation) -> Union[int, str]:
        return hash(value.json(exclude={"id"}))
