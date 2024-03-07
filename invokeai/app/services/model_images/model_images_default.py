from pathlib import Path

from PIL import Image
from PIL.Image import Image as PILImageType
from send2trash import send2trash

from invokeai.app.services.invoker import Invoker
from invokeai.app.util.misc import uuid_string
from invokeai.app.util.thumbnails import make_thumbnail

from .model_images_base import ModelImageFileStorageBase
from .model_images_common import (
    ModelImageFileDeleteException,
    ModelImageFileNotFoundException,
    ModelImageFileSaveException,
)


class ModelImageFileStorageDisk(ModelImageFileStorageBase):
    """Stores images on disk"""

    def __init__(self, model_images_folder: Path):
        self._model_images_folder = model_images_folder
        self._validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, model_key: str) -> PILImageType:
        try:
            path = self.get_path(model_key)

            if not self._validate_path(path):
                raise ModelImageFileNotFoundException

            return Image.open(path)
        except FileNotFoundError as e:
            raise ModelImageFileNotFoundException from e

    def save(self, image: PILImageType, model_key: str) -> None:
        try:
            self._validate_storage_folders()
            image_path = self._model_images_folder / (model_key + ".webp")
            thumbnail = make_thumbnail(image, 256)
            thumbnail.save(image_path, format="webp")

        except Exception as e:
            raise ModelImageFileSaveException from e

    def get_path(self, model_key: str) -> Path:
        path = self._model_images_folder / (model_key + ".webp")

        return path

    def get_url(self, model_key: str) -> str | None:
        path = self.get_path(model_key)
        if not self._validate_path(path):
            return

        url = self._invoker.services.urls.get_model_image_url(model_key)

        # The image file
        url += f"?{uuid_string()}"

        return url

    def delete(self, model_key: str) -> None:
        try:
            path = self.get_path(model_key)

            if not self._validate_path(path):
                raise ModelImageFileNotFoundException

            send2trash(path)

        except Exception as e:
            raise ModelImageFileDeleteException from e

    def _validate_path(self, path: Path) -> bool:
        """Validates the path given for an image."""
        return path.exists()

    def _validate_storage_folders(self) -> None:
        """Checks if the required folders exist and create them if they don't"""
        self._model_images_folder.mkdir(parents=True, exist_ok=True)
