from abc import ABC, abstractmethod
from base64 import urlsafe_b64encode
from glob import glob
import json
import os
from pathlib import Path
from queue import Queue
from typing import Dict, List, Union
from uuid import uuid4
from pydantic import BaseModel
from pydantic.fields import Field
from .invocation_context import InvocationContext


class ContextManagerABC(ABC):
    """Base context manager class"""
    @abstractmethod
    def get(self, context_id: str) -> Union[InvocationContext,None]:
        pass

    @abstractmethod
    def set(self, context: InvocationContext) -> None:
        pass

    @abstractmethod
    def create(self) -> InvocationContext:
        pass


# TODO: Consider making this generic to support more paginated result types in the future
class PaginatedContexts(BaseModel):
    items: List[str] = Field(description = "Context ids")
    page: int        = Field(description = "Current Page")
    pages: int       = Field(description = "Total number of pages")
    per_page: int    = Field(description = "Number of items per page")
    total: int       = Field(description = "Total number of items in result")


class DiskContextManager(ContextManagerABC):
    """An in-memory context manager"""
    __output_folder: str
    __cache: Dict[str, InvocationContext]
    __cache_ids: Queue # TODO: this is an incredibly naive cache
    __max_cache_size: int

    def __init__(self, output_folder: str):
        self.__output_folder = os.path.join(output_folder, 'contexts')
        self.__cache = dict()
        self.__cache_ids = Queue()
        self.__max_cache_size = 10 # TODO: get this from config

        Path(self.__output_folder).mkdir(parents=True, exist_ok=True)

    def list(self, page: int = 0, per_page: int = 10) -> PaginatedContexts:
        files = sorted(
            glob(os.path.join(self.__output_folder, "*.json")),
            key=os.path.getmtime,
            reverse=True,
        )
        count = len(files)

        startId = page * per_page
        pageCount = int(count / per_page) + 1
        endId = min(startId + per_page, count)
        items = [] if startId >= count else files[startId:endId]

        items = list(map(lambda f: Path(f).stem, items))

        return PaginatedContexts(
            items = items,
            page = page,
            pages = pageCount,
            per_page = per_page,
            total = count
        )

    def get(self, context_id: str) -> Union[InvocationContext,None]:
        cache_item = self.__get_cache(context_id)
        if cache_item:
            return cache_item

        filename = self.__get_filename(context_id)
        context = InvocationContext.parse_file(filename)
        self.__set_cache(context.id, context)
        return context
    
    def set(self, context: InvocationContext) -> None:
        self.__set_cache(context)
        filename = self.__get_filename(context.id)
        Path(filename).write_text(context.json())

    def create(self) -> InvocationContext:
        # TODO: consider using a provided id generator from services?
        context_id = urlsafe_b64encode(uuid4().bytes).decode("ascii")
        context = InvocationContext(
            id              = context_id,
            change_callback = self.__context_changed)
        self.set(context)
        return context

    def __context_changed(self, context: InvocationContext) -> None:
        self.set(context)

    def __get_cache(self, context_id: str) -> Union[InvocationContext,None]:
        return None if context_id not in self.__cache else self.__cache[context_id]

    def __set_cache(self, context: InvocationContext):
        if not context.id in self.__cache:
            self.__cache[context.id] = context
            self.__cache_ids.put(context.id) # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                del self.__cache[cache_id]

    def __get_filename(self, context_id: str) -> str:
        return os.path.join(self.__output_folder, f'{context_id}.json')
