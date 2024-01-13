# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, Union

from PIL import Image, PngImagePlugin
from PIL.Image import Image as PILImageType
from send2trash import send2trash

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from invokeai.app.util.thumbnails import get_thumbnail_name, make_thumbnail

from .image_files_base import ImageFileStorageBase
from .image_files_common import ImageFileDeleteException, ImageFileNotFoundException, ImageFileSaveException


class DiskImageFileStorage(ImageFileStorageBase):
    """Stores images on disk"""

    __output_folder: Path
    __cache_ids: Queue  # TODO: this is an incredibly naive cache
    __cache: Dict[Path, PILImageType]
    __max_cache_size: int
    __invoker: Invoker

    def __init__(self, output_folder: Union[str, Path]):
        self.__cache = {}
        self.__cache_ids = Queue()
        self.__max_cache_size = 10  # TODO: get this from config

        self.__output_folder: Path = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        # Validate required output folders at launch
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def get(self, image_name: str) -> PILImageType:
        try:
            image_path = self.get_path(image_name)

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
        image_name: str,
        metadata: Optional[MetadataField] = None,
        workflow: Optional[WorkflowWithoutID] = None,
        thumbnail_size: int = 256,
    ) -> None:
        try:
            self.__validate_storage_folders()
            image_path = self.get_path(image_name)

            pnginfo = PngImagePlugin.PngInfo()
            info_dict = {}

            if metadata is not None:
                metadata_json = metadata.model_dump_json()
                info_dict["invokeai_metadata"] = metadata_json
                pnginfo.add_text("invokeai_metadata", metadata_json)
            if workflow is not None:
                workflow_json = workflow.model_dump_json()
                info_dict["invokeai_workflow"] = workflow_json
                pnginfo.add_text("invokeai_workflow", workflow_json)

            # When saving the image, the image object's info field is not populated. We need to set it
            image.info = info_dict
            image.save(
                image_path,
                "PNG",
                pnginfo=pnginfo,
                compress_level=self.__invoker.services.configuration.png_compress_level,
            )

            thumbnail_name = get_thumbnail_name(image_name)
            thumbnail_path = self.get_path(thumbnail_name, thumbnail=True)
            thumbnail_image = make_thumbnail(image, thumbnail_size)
            thumbnail_image.save(thumbnail_path)

            self.__set_cache(image_path, image)
            self.__set_cache(thumbnail_path, thumbnail_image)
        except Exception as e:
            raise ImageFileSaveException from e

    def delete(self, image_name: str) -> None:
        try:
            image_path = self.get_path(image_name)

            if image_path.exists():
                send2trash(image_path)
            if image_path in self.__cache:
                del self.__cache[image_path]

            thumbnail_name = get_thumbnail_name(image_name)
            thumbnail_path = self.get_path(thumbnail_name, True)

            if thumbnail_path.exists():
                send2trash(thumbnail_path)
            if thumbnail_path in self.__cache:
                del self.__cache[thumbnail_path]
        except Exception as e:
            raise ImageFileDeleteException from e

    # TODO: make this a bit more flexible for e.g. cloud storage
    def get_path(self, image_name: str, thumbnail: bool = False) -> Path:
        path = self.__output_folder / image_name

        if thumbnail:
            thumbnail_name = get_thumbnail_name(image_name)
            path = self.__thumbnails_folder / thumbnail_name

        return path

    def validate_path(self, path: Union[str, Path]) -> bool:
        """Validates the path given for an image or thumbnail."""
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    def get_workflow(self, image_name: str) -> WorkflowWithoutID | None:
        image = self.get(image_name)
        workflow = image.info.get("invokeai_workflow", None)
        if workflow is not None:
            return WorkflowWithoutID.model_validate_json(workflow)
        return None

    def __validate_storage_folders(self) -> None:
        """Checks if the required output folders exist and create them if they don't"""
        folders: list[Path] = [self.__output_folder, self.__thumbnails_folder]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    def __get_cache(self, image_name: Path) -> Optional[PILImageType]:
        return None if image_name not in self.__cache else self.__cache[image_name]

    def __set_cache(self, image_name: Path, image: PILImageType):
        if image_name not in self.__cache:
            self.__cache[image_name] = image
            self.__cache_ids.put(image_name)  # TODO: this should refresh position for LRU cache
            if len(self.__cache) > self.__max_cache_size:
                cache_id = self.__cache_ids.get()
                if cache_id in self.__cache:
                    del self.__cache[cache_id]
