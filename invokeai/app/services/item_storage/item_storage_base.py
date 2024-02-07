from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class ItemStorageABC(ABC, Generic[T]):
    """Provides storage for a single type of item. The type must be a Pydantic model."""

    _on_changed_callbacks: list[Callable[[T], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = []
        self._on_deleted_callbacks = []

    """Base item storage class"""

    @abstractmethod
    def get(self, item_id: str) -> T:
        """
        Gets the item.
        :param item_id: the id of the item to get
        :raises ItemNotFoundError: if the item is not found
        """
        pass

    @abstractmethod
    def set(self, item: T) -> str:
        """
        Sets the item. The id will be extracted based on id_field.
        :param item: the item to set
        """
        pass

    @abstractmethod
    def delete(self, item_id: str) -> None:
        """
        Deletes the item, if it exists.
        """
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
