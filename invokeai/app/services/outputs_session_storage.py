from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

class PaginatedStringResults(GenericModel):
    """Paginated results"""
    #fmt: off
    items: list[str] = Field(description="Session IDs")
    page: int = Field(description="Current Page")
    pages: int = Field(description="Total number of pages")
    per_page: int = Field(description="Number of items per page")
    total: int = Field(description="Total number of items in result")
    #fmt: on

class OutputsSessionStorageABC(ABC):
    _on_changed_callbacks: list[Callable[[str], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = list()
        self._on_deleted_callbacks = list()

    """Base item storage class"""

    @abstractmethod
    def get(self, output_id: str) -> str:
        pass

    @abstractmethod
    def set(self, output_id: str, session_id: str) -> None:
        pass

    @abstractmethod
    def list(self, page: int = 0, per_page: int = 10) -> PaginatedStringResults:
        pass

    @abstractmethod
    def search(
        self, query: str, page: int = 0, per_page: int = 10
    ) -> PaginatedStringResults:
        pass

    def on_changed(self, on_changed: Callable[[str], None]) -> None:
        """Register a callback for when an item is changed"""
        self._on_changed_callbacks.append(on_changed)

    def on_deleted(self, on_deleted: Callable[[str], None]) -> None:
        """Register a callback for when an item is deleted"""
        self._on_deleted_callbacks.append(on_deleted)

    def _on_changed(self, foreign_key_value: str) -> None:
        for callback in self._on_changed_callbacks:
            callback(foreign_key_value)

    def _on_deleted(self, item_id: str) -> None:
        for callback in self._on_deleted_callbacks:
            callback(item_id)
