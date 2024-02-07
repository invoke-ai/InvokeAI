# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class PickleStorageBase(ABC, Generic[T]):
    """Responsible for storing and retrieving non-serializable data using a pickler."""

    _on_changed_callbacks: list[Callable[[T], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = []
        self._on_deleted_callbacks = []

    @abstractmethod
    def get(self, name: str) -> T:
        pass

    @abstractmethod
    def save(self, name: str, data: T) -> None:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
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
