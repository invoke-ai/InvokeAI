# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
import os
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from typing import Dict, Optional

from PIL.Image import Image as PILImageType
from PIL import Image, PngImagePlugin
from send2trash import send2trash

from invokeai.app.models.image import ResourceOrigin
from invokeai.app.models.metadata import ImageMetadata
from invokeai.app.util.thumbnails import get_thumbnail_name, make_thumbnail


# TODO: Should these excpetions subclass existing python exceptions?
class ImageFileNotFoundException(Exception):
    """Raised when an image file is not found in storage."""

    def __init__(self, message="Image file not found"):
        super().__init__(message)


class ImageFileSaveException(Exception):
    """Raised when an image cannot be saved."""

    def __init__(self, message="Image file not saved"):
        super().__init__(message)


class ImageFileDeleteException(Exception):
    """Raised when an image cannot be deleted."""

    def __init__(self, message="Image file not deleted"):
        super().__init__(message)


class ImageFileStorageBase(ABC):
    """Low-level service responsible for storing and retrieving image files."""

    @abstractmethod
    def get(self, image_origin: ResourceOrigin, image_name: str) -> PILImageType:
        """Retrieves an image as PIL Image."""
        pass

    @abstractmethod
    def get_path(
        self, image_origin: ResourceOrigin, image_name: str, thumbnail: bool = False
    ) -> str:
        """Gets the internal path to an image or thumbnail."""
        pass

    # TODO: We need to validate paths before starlette makes the FileResponse, else we get a
    # 500 internal server error. I don't like having this method on the service.
    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates the path given for an image or thumbnail."""
        pass

    @abstractmethod
    def save(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_name: str,
        metadata: Optional[ImageMetadata] = None,
        thumbnail_size: int = 256,
    ) -> None:
        """Saves an image and a 256x256 WEBP thumbnail. Returns a tuple of the image name, thumbnail name, and created timestamp."""
        pass

    @abstractmethod
    def delete(self, image_origin: ResourceOrigin, image_name: str) -> None:
        """Deletes an image and its thumbnail (if one exists)."""
        pass


class DiskImageFileStorage(ImageFileStorageBase):
    """Stores images on disk"""

    __output_folder: str
    __cache_ids: Queue  # TODO: this is an incredibly naive cache
    __cache: Dict[str, PILImageType]
    __max_cache_size: int

    def __init__(self, output_folder: str):
        self.__output_folder = output_folder
        self.__cache = dict()
        self.__cache_ids = Queue()
        self.__max_cache_size = 10  # TODO: get this from config

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # TODO: don't hard-code. get/save/delete should maybe take subpath?
        for image_origin in ResourceOrigin:
            Path(os.path.join(output_folder, image_origin)).mkdir(
                parents=True, exist_ok=True
            )
            Path(os.path.join(output_folder, image_origin, "thumbnails")).mkdir(
                parents=True, exist_ok=True
            )

    def get(self, image_origin: ResourceOrigin, image_name: str) -> PILImageType:
        try:
            image_path = self.get_path(image_origin, image_name)
            cache_item = self.__get_cache(image_path)
            if cache_item:
                return cache_item

            image = Image.open(image_path)
            self.__set_cache(image_path, image)
            return image
        except FileNotFoundError as e:
            raise ImageFileNotFoundException from e

    def save(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_name: str,
        metadata: Optional[ImageMetadata] = None,
        thumbnail_size: int = 256,
    ) -> None:
        try:
            image_path = self.get_path(image_origin, image_name)

            if metadata is not None:
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("invokeai", metadata.json())
                image.save(image_path, "PNG", pnginfo=pnginfo)
            else:
                image.save(image_path, "PNG")

            thumbnail_name = get_thumbnail_name(image_name)
            thumbnail_path = self.get_path(image_origin, thumbnail_name, thumbnail=True)
            thumbnail_image = make_thumbnail(image, thumbnail_size)
            thumbnail_image.save(thumbnail_path)

            self.__set_cache(image_path, image)
            self.__set_cache(thumbnail_path, thumbnail_image)
        except Exception as e:
            raise ImageFileSaveException from e

    def delete(self, image_origin: ResourceOrigin, image_name: str) -> None:
        try:
            basename = os.path.basename(image_name)
            image_path = self.get_path(image_origin, basename)

            if os.path.exists(image_path):
                send2trash(image_path)
            if image_path in self.__cache:
                del self.__cache[image_path]

            thumbnail_name = get_thumbnail_name(image_name)
            thumbnail_path = self.get_path(image_origin, thumbnail_name, True)

            if os.path.exists(thumbnail_path):
                send2trash(thumbnail_path)
            if thumbnail_path in self.__cache:
                del self.__cache[thumbnail_path]
        except Exception as e:
            raise ImageFileDeleteException from e

    # TODO: make this a bit more flexible for e.g. cloud storage
    def get_path(
        self, image_origin: ResourceOrigin, image_name: str, thumbnail: bool = False
    ) -> str:
        # strip out any relative path shenanigans
        basename = os.path.basename(image_name)

        if thumbnail:
            thumbnail_name = get_thumbnail_name(basename)
            path = os.path.join(
                self.__output_folder, image_origin, "thumbnails", thumbnail_name
            )
        else:
            path = os.path.join(self.__output_folder, image_origin, basename)

        abspath = os.path.abspath(path)

        return abspath

    def validate_path(self, path: str) -> bool:
        """Validates the path given for an image or thumbnail."""
        try:
            os.stat(path)
            return True
        except:
            return False

    def __get_cache(self, image_name: str) -> PILImageType | None:
        return None if image_name not in self.__cache else self.__cache[image_name]

    def __set_cache(self, image_name: str, image: PILImageType):
        if not image_name in self.__cache:
            self.__cache[image_name] = image
            self.__cache_ids.put(
                image_name
            )  # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                if cache_id in self.__cache:
                    del self.__cache[cache_id]
