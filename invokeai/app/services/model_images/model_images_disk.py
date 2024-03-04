from pathlib import Path
from typing import Union

from PIL import Image, PngImagePlugin
from PIL.Image import Image as PILImageType
from send2trash import send2trash

from invokeai.app.services.invoker import Invoker

from .model_images_base import ModelImagesBase
from .model_images_common import ModelImageFileDeleteException, ModelImageFileNotFoundException, ModelImageFileSaveException


class DiskImageFileStorage(ModelImagesBase):
    """Stores images on disk"""

    __output_folder: Path
    __invoker: Invoker

    def __init__(self, output_folder: Union[str, Path]):

        self.__output_folder: Path = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        # Validate required output folders at launch
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def get(self, image_name: str) -> PILImageType:
        try:
            image_path = self.get_path(image_name)

            image = Image.open(image_path)
            return image
        except FileNotFoundError as e:
            raise ModelImageFileNotFoundException from e

    def save(
        self,
        image: PILImageType,
        image_name: str,
    ) -> None:
        try:
            self.__validate_storage_folders()
            image_path = self.get_path(image_name)

            pnginfo = PngImagePlugin.PngInfo()
            info_dict = {}

            # When saving the image, the image object's info field is not populated. We need to set it
            image.info = info_dict
            image.save(
                image_path,
                "PNG",
                pnginfo=pnginfo,
                compress_level=self.__invoker.services.configuration.png_compress_level,
            )

        except Exception as e:
            raise ModelImageFileSaveException from e

    def delete(self, image_name: str) -> None:
        try:
            image_path = self.get_path(image_name)

            if image_path.exists():
                send2trash(image_path)

        except Exception as e:
            raise ModelImageFileDeleteException from e

    # TODO: make this a bit more flexible for e.g. cloud storage
    def get_path(self, image_name: str) -> Path:
        path = self.__output_folder / image_name

        return path

    def __validate_storage_folders(self) -> None:
        """Checks if the required output folders exist and create them if they don't"""
        folders: list[Path] = [self.__output_folder]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
