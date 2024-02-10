# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from pathlib import Path
from typing import Union

import torch

from invokeai.app.invocations.compel import ConditioningFieldData
from invokeai.app.services.invoker import Invoker

from .latents_storage_base import LatentsStorageBase


class DiskLatentsStorage(LatentsStorageBase):
    """Stores latents in a folder on disk without caching"""

    __output_folder: Path

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__output_folder.mkdir(parents=True, exist_ok=True)

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._delete_all_latents()

    def get(self, name: str) -> torch.Tensor:
        latent_path = self.get_path(name)
        return torch.load(latent_path)

    def save(self, name: str, data: Union[torch.Tensor, ConditioningFieldData]) -> None:
        self.__output_folder.mkdir(parents=True, exist_ok=True)
        latent_path = self.get_path(name)
        torch.save(data, latent_path)

    def delete(self, name: str) -> None:
        latent_path = self.get_path(name)
        latent_path.unlink()

    def get_path(self, name: str) -> Path:
        return self.__output_folder / name

    def _delete_all_latents(self) -> None:
        """
        Deletes all latents from disk.
        Must be called after we have access to `self._invoker` (e.g. in `start()`).
        """
        deleted_latents_count = 0
        freed_space = 0
        for latents_file in Path(self.__output_folder).glob("*"):
            if latents_file.is_file():
                freed_space += latents_file.stat().st_size
                deleted_latents_count += 1
                latents_file.unlink()
        if deleted_latents_count > 0:
            freed_space_in_mb = round(freed_space / 1024 / 1024, 2)
            self._invoker.services.logger.info(
                f"Deleted {deleted_latents_count} latents files (freed {freed_space_in_mb}MB)"
            )
