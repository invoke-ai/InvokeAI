# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from calendar import c
from enum import Enum
import os
from pathlib import Path
from queue import Queue
from typing import Dict
from PIL.Image import Image
from ...pngwriter import PngWriter


class ImageType(str, Enum):
    RESULT = 'results'
    INTERMEDIATE = 'intermediates'


class ImageStorageBase(ABC):
    """Responsible for storing and retrieving images."""

    @abstractmethod
    def get(self, image_type: ImageType, image_name: str) -> Image:
        pass

    # TODO: make this a bit more flexible for e.g. cloud storage
    @abstractmethod
    def get_path(self, image_type: ImageType, image_name: str) -> str:
        pass

    @abstractmethod
    def save(self, image_type: ImageType, image_name: str, image: Image) -> None:
        pass

    @abstractmethod
    def delete(self, image_type: ImageType, image_name: str) -> None:
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
        for image_type in ImageType:
            Path(os.path.join(output_folder, image_type)).mkdir(parents=True, exist_ok=True)

    def get(self, image_type: ImageType, image_name: str) -> Image:
        image_path = self.get_path(image_type, image_name)
        cache_item = self.__get_cache(image_path)
        if cache_item:
            return cache_item

        image = Image.open(image_path)
        self.__set_cache(image_path, image)
        return image

    # TODO: make this a bit more flexible for e.g. cloud storage
    def get_path(self, image_type: ImageType, image_name: str) -> str:
        path = os.path.join(self.__output_folder, image_type, image_name)
        return path

    def save(self, image_type: ImageType, image_name: str, image: Image) -> None:
        image_subpath = os.path.join(image_type, image_name)
        self.__pngWriter.save_image_and_prompt_to_png(image, "", image_subpath, None) # TODO: just pass full path to png writer

        image_path = self.get_path(image_type, image_name)
        self.__set_cache(image_path, image)

    def delete(self, image_type: ImageType, image_name: str) -> None:
        image_path = self.get_path(image_type, image_name)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        if image_path in self.__cache:
            del self.__cache[image_path]

    def __get_cache(self, image_name: str) -> Image:
        return None if image_name not in self.__cache else self.__cache[image_name]

    def __set_cache(self, image_name: str, image: Image):
        if not image_name in self.__cache:
            self.__cache[image_name] = image
            self.__cache_ids.put(image_name) # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                del self.__cache[cache_id]
