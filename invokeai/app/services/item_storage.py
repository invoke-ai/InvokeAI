from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar("T", bound=BaseModel)


class PaginatedResults(GenericModel, Generic[T]):
    """Paginated results"""

    # fmt: off
    items: list[T] = Field(description="Items")
    page: int = Field(description="Current Page")
    pages: int = Field(description="Total number of pages")
    per_page: int = Field(description="Number of items per page")
    total: int = Field(description="Total number of items in result")
    # fmt: on


class ItemStorageABC(ABC, Generic[T]):
    _on_changed_callbacks: list[Callable[[T], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = list()
        self._on_deleted_callbacks = list()

    """Base item storage class"""

    @abstractmethod
    def get(self, item_id: str) -> T:
        """Gets the item, parsing it into a Pydantic model"""
        pass

    @abstractmethod
    def get_raw(self, item_id: str) -> Optional[str]:
        """Gets the raw item as a string, skipping Pydantic parsing"""
        pass

    @abstractmethod
    def set(self, item: T) -> None:
        """Sets the item"""
        pass

    @abstractmethod
    def list(self, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        """Gets a paginated list of items"""
        pass

    @abstractmethod
    def search(self, query: str, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        pass

    def on_changed(self, on_changed: Callable[[T], None]) -> None:
        """Register a callback for when an item is changed"""
        self._on_changed_callbacks.append(on_changed)

    def on_deleted(self, on_deleted: Callable[[str], None]) -> None:
        """Register a callback for when an item is deleted"""
        self._on_deleted_callbacks.append(on_deleted)

    def _on_changed(self, item: T) -> None:
        for callback in self._on_changed_callbacks:
            callback(item)

    def _on_deleted(self, item_id: str) -> None:
        for callback in self._on_deleted_callbacks:
            callback(item_id)
