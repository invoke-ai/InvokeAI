import typing
from pathlib import Path
from typing import Optional, TypeVar

import torch

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.item_storage.item_storage_base import ItemStorageABC
from invokeai.app.util.misc import uuid_string

T = TypeVar("T")


class ItemStorageEphemeralDisk(ItemStorageABC[T]):
    """Provides arbitrary item storage with a disk-backed ephemeral storage. The storage is cleared at startup."""

    def __init__(self, output_folder: Path):
        super().__init__()
        self._output_folder = output_folder
        self._output_folder.mkdir(parents=True, exist_ok=True)
        self.__item_class_name: Optional[str] = None

    @property
    def _item_class_name(self) -> str:
        if not self.__item_class_name:
            # `__orig_class__` is not available in the constructor for some technical, undoubtedly very pythonic reason
            self.__item_class_name = typing.get_args(self.__orig_class__)[0].__name__  # pyright: ignore [reportUnknownMemberType, reportGeneralTypeIssues]
        return self.__item_class_name

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._delete_all_items()

    def get(self, item_id: str) -> T:
        file_path = self._get_path(item_id)
        return torch.load(file_path)  # pyright: ignore [reportUnknownMemberType]

    def set(self, item: T) -> str:
        self._output_folder.mkdir(parents=True, exist_ok=True)
        item_id = f"{self._item_class_name}_{uuid_string()}"
        file_path = self._get_path(item_id)
        torch.save(item, file_path)  # pyright: ignore [reportUnknownMemberType]
        return item_id

    def delete(self, item_id: str) -> None:
        file_path = self._get_path(item_id)
        file_path.unlink()

    def _get_path(self, item_id: str) -> Path:
        return self._output_folder / item_id

    def _delete_all_items(self) -> None:
        """
        Deletes all pickled items from disk.
        Must be called after we have access to `self._invoker` (e.g. in `start()`).
        """

        if not self._invoker:
            raise ValueError("Invoker is not set. Must call `start()` first.")

        deleted_count = 0
        freed_space = 0
        for file in Path(self._output_folder).glob("*"):
            if file.is_file():
                freed_space += file.stat().st_size
                deleted_count += 1
                file.unlink()
        if deleted_count > 0:
            freed_space_in_mb = round(freed_space / 1024 / 1024, 2)
            self._invoker.services.logger.info(
                f"Deleted {deleted_count} {self._item_class_name} files (freed {freed_space_in_mb}MB)"
            )
