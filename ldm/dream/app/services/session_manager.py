from abc import ABC, abstractmethod
from base64 import urlsafe_b64encode
from glob import glob
import os
from pathlib import Path
from queue import Queue
from typing import Dict, List, Union
from uuid import uuid4
from pydantic import BaseModel
from pydantic.fields import Field
from .invocation_session import InvocationSession


class SessionManagerABC(ABC):
    """Base session manager class"""
    @abstractmethod
    def get(self, session_id: str) -> Union[InvocationSession,None]:
        pass

    @abstractmethod
    def set(self, session: InvocationSession) -> None:
        pass

    @abstractmethod
    def create(self) -> InvocationSession:
        pass


# TODO: Consider making this generic to support more paginated result types in the future
class PaginatedSession(BaseModel):
    items: List[str] = Field(description = "Session ids")
    page: int        = Field(description = "Current Page")
    pages: int       = Field(description = "Total number of pages")
    per_page: int    = Field(description = "Number of items per page")
    total: int       = Field(description = "Total number of items in result")


class DiskSessionManager(SessionManagerABC):
    """An in-memory session manager"""
    __output_folder: str
    __cache: Dict[str, InvocationSession]
    __cache_ids: Queue # TODO: this is an incredibly naive cache
    __max_cache_size: int

    def __init__(self, output_folder: str):
        self.__output_folder = os.path.join(output_folder, 'sessions')
        self.__cache = dict()
        self.__cache_ids = Queue()
        self.__max_cache_size = 10 # TODO: get this from config

        Path(self.__output_folder).mkdir(parents=True, exist_ok=True)

    def list(self, page: int = 0, per_page: int = 10) -> PaginatedSession:
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

        return PaginatedSession(
            items = items,
            page = page,
            pages = pageCount,
            per_page = per_page,
            total = count
        )

    def get(self, session_id: str) -> Union[InvocationSession,None]:
        cache_item = self.__get_cache(session_id)
        if cache_item:
            return cache_item

        filename = self.__get_filename(session_id)
        session = InvocationSession.parse_file(filename)
        self.__set_cache(session.id, session)
        return session
    
    def set(self, session: InvocationSession) -> None:
        self.__set_cache(session)
        filename = self.__get_filename(session.id)
        Path(filename).write_text(session.json())

    def create(self) -> InvocationSession:
        # TODO: consider using a provided id generator from services?
        session_id = urlsafe_b64encode(uuid4().bytes).decode("ascii")
        session = InvocationSession(
            id              = session_id,
            change_callback = self.__session_changed)
        self.set(session)
        return session

    def __session_changed(self, session: InvocationSession) -> None:
        self.set(session)

    def __get_cache(self, session_id: str) -> Union[InvocationSession,None]:
        return None if session_id not in self.__cache else self.__cache[session_id]

    def __set_cache(self, session: InvocationSession):
        if not session.id in self.__cache:
            self.__cache[session.id] = session
            self.__cache_ids.put(session.id) # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                del self.__cache[cache_id]

    def __get_filename(self, session_id: str) -> str:
        return os.path.join(self.__output_folder, f'{session_id}.json')
