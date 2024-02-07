from pathlib import Path
from typing import Callable, TypeAlias, TypeVar


class ItemNotFoundError(KeyError):
    """Raised when an item is not found in storage"""

    def __init__(self, item_id: str) -> None:
        super().__init__(f"Item with id {item_id} not found")


T = TypeVar("T")

SaveFunc: TypeAlias = Callable[[T, Path], None]
LoadFunc: TypeAlias = Callable[[Path], T]
