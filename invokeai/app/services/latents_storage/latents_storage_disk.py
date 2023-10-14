# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from pathlib import Path
from typing import Union

import torch

from .latents_storage_base import LatentsStorageBase


class DiskLatentsStorage(LatentsStorageBase):
    """Stores latents in a folder on disk without caching"""

    __output_folder: Path

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__output_folder.mkdir(parents=True, exist_ok=True)

    def get(self, name: str) -> torch.Tensor:
        latent_path = self.get_path(name)
        return torch.load(latent_path)

    def save(self, name: str, data: torch.Tensor) -> None:
        self.__output_folder.mkdir(parents=True, exist_ok=True)
        latent_path = self.get_path(name)
        torch.save(data, latent_path)

    def delete(self, name: str) -> None:
        latent_path = self.get_path(name)
        latent_path.unlink()

    def get_path(self, name: str) -> Path:
        return self.__output_folder / name
