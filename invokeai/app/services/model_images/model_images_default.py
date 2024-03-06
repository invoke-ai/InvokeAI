from pathlib import Path
from typing import Union

from PIL import Image, PngImagePlugin
from PIL.Image import Image as PILImageType
from send2trash import send2trash

from invokeai.app.services.invoker import Invoker

from .model_images_base import ModelImagesBase
from .model_images_common import ModelImageFileDeleteException, ModelImageFileNotFoundException, ModelImageFileSaveException

class ModelImagesService(ModelImagesBase):
    """Stores images on disk"""

    __model_images_folder: Path
    __invoker: Invoker

    def __init__(self, model_images_folder: Union[str, Path]):

        self.__model_images_folder: Path = model_images_folder if isinstance(model_images_folder, Path) else Path(model_images_folder)
        # Validate required folders at launch
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def get(self, model_key: str) -> PILImageType:
        try:
            image_path = self.get_path(model_key + '.png')

            image = Image.open(image_path)
            return image
        except FileNotFoundError as e:
            raise ModelImageFileNotFoundException from e

    def save(
        self,
        image: PILImageType,
        model_key: str,
    ) -> None:
        try:
            self.__validate_storage_folders()
            image_path = self.get_path(model_key + '.png')
            pnginfo = PngImagePlugin.PngInfo()

            image.save(
                image_path,
                "PNG",
                pnginfo=pnginfo,
                compress_level=self.__invoker.services.configuration.png_compress_level,
            )

        except Exception as e:
            raise ModelImageFileSaveException from e

    def get_path(self, model_key: str) -> Path:
        path = self.__model_images_folder / model_key

        return path
    
    def get_url(self, model_key: str) -> str:
        return self.__invoker.services.urls.get_model_image_url(model_key)
    
    def delete(self, model_key: str) -> None:
        try:
            image_path = self.get_path(model_key + '.png')

            if image_path.exists():
                send2trash(image_path)

        except Exception as e:
            raise ModelImageFileDeleteException from e

    def __validate_storage_folders(self) -> None:
        """Checks if the required folders exist and create them if they don't"""
        folders: list[Path] = [self.__model_images_folder]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
