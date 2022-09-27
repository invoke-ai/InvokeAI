# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from calendar import c
import os
from pathlib import Path
from queue import Queue
from typing import Dict
from PIL.Image import Image
from ...pngwriter import PngWriter


class ImageStorageBase(ABC):
    """Responsible for storing and retrieving images."""

    @abstractmethod
    def get(self, image_uri: str) -> Image:
        pass

    @abstractmethod
    def save(self, image_uri: str, image: Image) -> None:
        pass

    @abstractmethod
    def delete(self, image_uri: str) -> None:
        pass


class DiskImageStorage(ImageStorageBase):
    """Stores images on disk"""
    __output_folder: str
    __pngWriter: PngWriter
    __cache_ids: Queue # TODO: this is an incredibly naive cache
    __cache: Dict[str, Image]
    __max_cache_size: int

    def __init__(self, output_folder: str):
        self.__output_folder = output_folder
        self.__pngWriter = PngWriter(output_folder)
        self.__cache = dict()
        self.__cache_ids = Queue()
        self.__max_cache_size = 10 # TODO: get this from config

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # TODO: don't hard-code. get/save/delete should maybe take subpath?
        Path(os.path.join(output_folder, '/results')).mkdir(parents=True, exist_ok=True)

    def get(self, image_uri: str) -> Image:
        cache_item = self.__get_cache(image_uri)
        if cache_item:
            return cache_item

        image = Image.open(image_uri)
        self.__set_cache(image_uri, image)
        return image

    def save(self, image_uri: str, image: Image) -> None:
        self.__pngWriter.save_image_and_prompt_to_png(image, "", image_uri, None)
        self.__set_cache(image_uri, image)

    def delete(self, image_uri: str) -> None:
        path = os.path.join(self.__output_folder, image_uri)
        if os.path.exists(path):
            os.remove(path)
        
        if image_uri in self.__cache:
            del self.__cache[image_uri]

    def __get_cache(self, image_uri: str) -> Image:
        return None if image_uri not in self.__cache else self.__cache[image_uri]

    def __set_cache(self, image_uri: str, image: Image):
        if not image_uri in self.__cache:
            self.__cache[image_uri] = image
            self.__cache_ids.put(image_uri) # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                del self.__cache[cache_id]
