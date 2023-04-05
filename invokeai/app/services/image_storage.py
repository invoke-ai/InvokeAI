# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import datetime
import os
from glob import glob
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, List

from PIL.Image import Image
import PIL.Image as PILImage
from pydantic import BaseModel
from invokeai.app.datatypes.image import ImageField, ImageResponse, ImageType
from invokeai.app.datatypes.metadata import ImageMetadata
from invokeai.app.services.item_storage import PaginatedResults
from invokeai.app.util.save_thumbnail import save_thumbnail

from invokeai.backend.image_util import PngWriter


class ImageStorageBase(ABC):
    """Responsible for storing and retrieving images."""

    @abstractmethod
    def get(self, image_type: ImageType, image_name: str) -> Image:
        pass

    @abstractmethod
    def list(
        self, image_type: ImageType, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ImageResponse]:
        pass

    # TODO: make this a bit more flexible for e.g. cloud storage
    @abstractmethod
    def get_path(
        self, image_type: ImageType, image_name: str, is_thumbnail: bool = False
    ) -> str:
        pass

    @abstractmethod
    def save(self, image_type: ImageType, image_name: str, image: Image) -> None:
        pass

    @abstractmethod
    def delete(self, image_type: ImageType, image_name: str) -> None:
        pass

    def create_name(self, context_id: str, node_id: str) -> str:
        return f"{context_id}_{node_id}_{str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))}.png"


class DiskImageStorage(ImageStorageBase):
    """Stores images on disk"""

    __output_folder: str
    __pngWriter: PngWriter
    __cache_ids: Queue  # TODO: this is an incredibly naive cache
    __cache: Dict[str, Image]
    __max_cache_size: int

    def __init__(self, output_folder: str):
        self.__output_folder = output_folder
        self.__pngWriter = PngWriter(output_folder)
        self.__cache = dict()
        self.__cache_ids = Queue()
        self.__max_cache_size = 10  # TODO: get this from config

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # TODO: don't hard-code. get/save/delete should maybe take subpath?
        for image_type in ImageType:
            Path(os.path.join(output_folder, image_type)).mkdir(
                parents=True, exist_ok=True
            )
            Path(os.path.join(output_folder, image_type, "thumbnails")).mkdir(
                parents=True, exist_ok=True
            )

    def list(
        self, image_type: ImageType, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ImageResponse]:
        dir_path = os.path.join(self.__output_folder, image_type)
        image_paths = glob(f"{dir_path}/*.png")
        count = len(image_paths)

        # TODO: do all platforms support `getmtime`? seem to recall some do not...
        sorted_image_paths = sorted(
            glob(f"{dir_path}/*.png"), key=os.path.getmtime, reverse=True
        )

        page_of_image_paths = sorted_image_paths[
            page * per_page : (page + 1) * per_page
        ]

        page_of_images: List[ImageResponse] = []

        for path in page_of_image_paths:
            filename = os.path.basename(path)
            img = PILImage.open(path)
            page_of_images.append(
                ImageResponse(
                    image_type=image_type.value,
                    image_name=os.path.basename(path),
                    # TODO: DiskImageStorage should not be building URLs...?
                    image_url=f"api/v1/images/{image_type.value}/{filename}",
                    thumbnail_url=f"api/v1/images/{image_type.value}/thumbnails/{os.path.splitext(filename)[0]}.webp",
                    # TODO: Creation of this object should happen elsewhere, just making it fit here so it works
                    metadata=ImageMetadata(
                        timestamp=int(os.path.splitext(filename)[0].split("_")[-1]),
                        width=img.width,
                        height=img.height,
                    ),
                )
            )

        page_count_trunc = int(count / per_page)
        page_count_mod = count % per_page
        page_count = page_count_trunc if page_count_mod == 0 else page_count_trunc + 1

        return PaginatedResults[ImageResponse](
            items=page_of_images,
            page=page,
            pages=page_count,
            per_page=per_page,
            total=count,
        )

    def get(self, image_type: ImageType, image_name: str) -> Image:
        image_path = self.get_path(image_type, image_name)
        cache_item = self.__get_cache(image_path)
        if cache_item:
            return cache_item

        image = PILImage.open(image_path)
        self.__set_cache(image_path, image)
        return image

    # TODO: make this a bit more flexible for e.g. cloud storage
    def get_path(
        self, image_type: ImageType, image_name: str, is_thumbnail: bool = False
    ) -> str:
        if is_thumbnail:
            path = os.path.join(
                self.__output_folder, image_type, "thumbnails", image_name
            )
        else:
            path = os.path.join(self.__output_folder, image_type, image_name)
        return path

    def save(self, image_type: ImageType, image_name: str, image: Image) -> None:
        image_subpath = os.path.join(image_type, image_name)
        self.__pngWriter.save_image_and_prompt_to_png(
            image, "", image_subpath, None
        )  # TODO: just pass full path to png writer
        save_thumbnail(
            image=image,
            filename=image_name,
            path=os.path.join(self.__output_folder, image_type, "thumbnails"),
        )
        image_path = self.get_path(image_type, image_name)
        self.__set_cache(image_path, image)

    def delete(self, image_type: ImageType, image_name: str) -> None:
        image_path = self.get_path(image_type, image_name)
        thumbnail_path = self.get_path(image_type, image_name, True)
        if os.path.exists(image_path):
            os.remove(image_path)

        if image_path in self.__cache:
            del self.__cache[image_path]

        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)

        if thumbnail_path in self.__cache:
            del self.__cache[thumbnail_path]

    def __get_cache(self, image_name: str) -> Image:
        return None if image_name not in self.__cache else self.__cache[image_name]

    def __set_cache(self, image_name: str, image: Image):
        if not image_name in self.__cache:
            self.__cache[image_name] = image
            self.__cache_ids.put(
                image_name
            )  # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                del self.__cache[cache_id]
