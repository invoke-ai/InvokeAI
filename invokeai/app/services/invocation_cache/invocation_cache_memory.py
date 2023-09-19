from queue import Queue
from typing import Optional, Union


from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
from invokeai.app.services.invoker import Invoker


class MemoryInvocationCache(InvocationCacheBase):
    __cache: dict[Union[int, str], tuple[BaseInvocationOutput, str]]
    __max_cache_size: int
    __cache_ids: Queue
    __invoker: Invoker

    def __init__(self, max_cache_size: int = 512) -> None:
        self.__cache = dict()
        self.__max_cache_size = max_cache_size
        self.__cache_ids = Queue()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self.__invoker.services.images.on_deleted(self.delete_by_match)

    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        if self.__max_cache_size == 0:
            return None

        item = self.__cache.get(key, None)
        if item is not None:
            return item[0]

    def save(self, value: BaseInvocationOutput) -> None:
        if self.__max_cache_size == 0:
            return None

        value_json = value.json(exclude={"id"})
        key = hash(value_json)

        if key not in self.__cache:
            self.__cache[key] = (value, value_json)
            self.__cache_ids.put(key)
            if self.__cache_ids.qsize() > self.__max_cache_size:
                try:
                    self.__cache.pop(self.__cache_ids.get())
                except KeyError:
                    pass

    def delete(self, key: Union[int, str]) -> None:
        if self.__max_cache_size == 0:
            return None

        if key in self.__cache:
            del self.__cache[key]

    def delete_by_match(self, to_match: str) -> None:
        to_delete = []
        for name, item in self.__cache.items():
            if to_match in item[1]:
                to_delete.append(name)
        for key in to_delete:
            self.delete(key)

    def clear(self, *args, **kwargs) -> None:
        self.__cache.clear()
        self.__cache_ids = Queue()

    @classmethod
    def create_key(cls, value: BaseInvocation) -> Union[int, str]:
        return hash(value.json(exclude={"id"}))
