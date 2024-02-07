# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from pathlib import Path
from typing import TypeVar

import torch

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.pickle_storage.pickle_storage_base import PickleStorageBase

T = TypeVar("T")


class PickleStorageTorch(PickleStorageBase[T]):
    """Responsible for storing and retrieving non-serializable data using `torch.save` and `torch.load`."""

    def __init__(self, output_folder: Path, item_type_name: "str"):
        super().__init__()
        self._output_folder = output_folder
        self._output_folder.mkdir(parents=True, exist_ok=True)
        self._item_type_name = item_type_name

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._delete_all_items()

    def get(self, name: str) -> T:
        latent_path = self._get_path(name)
        return torch.load(latent_path)

    def save(self, name: str, data: T) -> None:
        self._output_folder.mkdir(parents=True, exist_ok=True)
        latent_path = self._get_path(name)
        torch.save(data, latent_path)

    def delete(self, name: str) -> None:
        latent_path = self._get_path(name)
        latent_path.unlink()

    def _get_path(self, name: str) -> Path:
        return self._output_folder / name

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
                f"Deleted {deleted_count} {self._item_type_name} files (freed {freed_space_in_mb}MB)"
            )
