# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from typing import Dict, Union, Optional

import torch


class LatentsStorageBase(ABC):
    """Responsible for storing and retrieving latents."""

    @abstractmethod
    def get(self, name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def save(self, name: str, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass


class ForwardCacheLatentsStorage(LatentsStorageBase):
    """Caches the latest N latents in memory, writing-thorugh to and reading from underlying storage"""

    __cache: Dict[str, torch.Tensor]
    __cache_ids: Queue
    __max_cache_size: int
    __underlying_storage: LatentsStorageBase

    def __init__(self, underlying_storage: LatentsStorageBase, max_cache_size: int = 20):
        self.__underlying_storage = underlying_storage
        self.__cache = dict()
        self.__cache_ids = Queue()
        self.__max_cache_size = max_cache_size

    def get(self, name: str) -> torch.Tensor:
        cache_item = self.__get_cache(name)
        if cache_item is not None:
            return cache_item

        latent = self.__underlying_storage.get(name)
        self.__set_cache(name, latent)
        return latent

    def save(self, name: str, data: torch.Tensor) -> None:
        self.__underlying_storage.save(name, data)
        self.__set_cache(name, data)

    def delete(self, name: str) -> None:
        self.__underlying_storage.delete(name)
        if name in self.__cache:
            del self.__cache[name]

    def __get_cache(self, name: str) -> Optional[torch.Tensor]:
        return None if name not in self.__cache else self.__cache[name]

    def __set_cache(self, name: str, data: torch.Tensor):
        if not name in self.__cache:
            self.__cache[name] = data
            self.__cache_ids.put(name)
            if self.__cache_ids.qsize() > self.__max_cache_size:
                self.__cache.pop(self.__cache_ids.get())


class DiskLatentsStorage(LatentsStorageBase):
    """Stores latents in a folder on disk without caching"""

    __output_folder: Union[str, Path]

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
