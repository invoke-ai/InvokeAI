# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from queue import Queue
from typing import Dict, Optional

import torch

from .latents_storage_base import LatentsStorageBase


class ForwardCacheLatentsStorage(LatentsStorageBase):
    """Caches the latest N latents in memory, writing-thorugh to and reading from underlying storage"""

    __cache: Dict[str, torch.Tensor]
    __cache_ids: Queue
    __max_cache_size: int
    __underlying_storage: LatentsStorageBase

    def __init__(self, underlying_storage: LatentsStorageBase, max_cache_size: int = 20):
        super().__init__()
        self.__underlying_storage = underlying_storage
        self.__cache = {}
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
        self._on_changed(data)

    def delete(self, name: str) -> None:
        self.__underlying_storage.delete(name)
        if name in self.__cache:
            del self.__cache[name]
        self._on_deleted(name)

    def __get_cache(self, name: str) -> Optional[torch.Tensor]:
        return None if name not in self.__cache else self.__cache[name]

    def __set_cache(self, name: str, data: torch.Tensor):
        if name not in self.__cache:
            self.__cache[name] = data
            self.__cache_ids.put(name)
            if self.__cache_ids.qsize() > self.__max_cache_size:
                self.__cache.pop(self.__cache_ids.get())
