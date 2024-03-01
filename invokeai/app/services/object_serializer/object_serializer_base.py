from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class ObjectSerializerBase(ABC, Generic[T]):
    """Saves and loads arbitrary python objects."""

    def __init__(self) -> None:
        self._on_deleted_callbacks: list[Callable[[str], None]] = []

    @abstractmethod
    def load(self, name: str) -> T:
        """
        Loads the object.
        :param name: The name of the object to load.
        :raises ObjectNotFoundError: if the object is not found
        """
        pass

    @abstractmethod
    def save(self, obj: T) -> str:
        """
        Saves the object, returning its name.
        :param obj: The object to save.
        """
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        """
        Deletes the object, if it exists.
        :param name: The name of the object to delete.
        """
        pass

    def on_deleted(self, on_deleted: Callable[[str], None]) -> None:
        """Register a callback for when an object is deleted"""
        self._on_deleted_callbacks.append(on_deleted)

    def _on_deleted(self, name: str) -> None:
        for callback in self._on_deleted_callbacks:
            callback(name)
