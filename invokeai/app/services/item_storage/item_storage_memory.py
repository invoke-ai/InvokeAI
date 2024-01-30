from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

from invokeai.app.services.item_storage.item_storage_base import ItemStorageABC
from invokeai.app.services.shared.pagination import PaginatedResults

T = TypeVar("T", bound=BaseModel)


class ItemStorageMemory(ItemStorageABC, Generic[T]):
    def __init__(self, id_field: str = "id") -> None:
        super().__init__()
        self._id_field = id_field
        self._items: dict[str, T] = {}

    def get(self, item_id: str) -> Optional[T]:
        return self._items.get(item_id)

    def set(self, item: T) -> None:
        self._items[getattr(item, self._id_field)] = item
        self._on_changed(item)

    def delete(self, item_id: str) -> None:
        try:
            del self._items[item_id]
            self._on_deleted(item_id)
        except KeyError:
            pass

    def list(self, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        # TODO: actually paginate?
        return PaginatedResults(
            items=list(self._items.values()), page=page, per_page=per_page, pages=1, total=len(self._items)
        )

    def search(self, query: str, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        # TODO: actually paginate?
        # TODO: actually search?
        return PaginatedResults(
            items=list(self._items.values()), page=page, per_page=per_page, pages=1, total=len(self._items)
        )
