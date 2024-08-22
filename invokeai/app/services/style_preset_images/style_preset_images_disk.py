from pathlib import Path

from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.style_preset_images.style_preset_images_base import StylePresetImageFileStorageBase
from invokeai.app.services.style_preset_images.style_preset_images_common import (
    StylePresetImageFileDeleteException,
    StylePresetImageFileNotFoundException,
    StylePresetImageFileSaveException,
)
from invokeai.app.services.style_preset_records.style_preset_records_common import PresetType
from invokeai.app.util.misc import uuid_string
from invokeai.app.util.thumbnails import make_thumbnail


class StylePresetImageFileStorageDisk(StylePresetImageFileStorageBase):
    """Stores images on disk"""

    def __init__(self, style_preset_images_folder: Path):
        self._style_preset_images_folder = style_preset_images_folder
        self._validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, style_preset_id: str) -> PILImageType:
        try:
            path = self.get_path(style_preset_id)

            return Image.open(path)
        except FileNotFoundError as e:
            raise StylePresetImageFileNotFoundException from e

    def save(self, style_preset_id: str, image: PILImageType) -> None:
        try:
            self._validate_storage_folders()
            image_path = self._style_preset_images_folder / (style_preset_id + ".webp")
            thumbnail = make_thumbnail(image, 256)
            thumbnail.save(image_path, format="webp")

        except Exception as e:
            raise StylePresetImageFileSaveException from e

    def get_path(self, style_preset_id: str) -> Path:
        style_preset = self._invoker.services.style_preset_records.get(style_preset_id)
        if style_preset.type is PresetType.Default:
            default_images_dir = Path(__file__).parent / Path("default_style_preset_images")
            path = default_images_dir / (style_preset.name + ".png")
        else:
            path = self._style_preset_images_folder / (style_preset_id + ".webp")

        return path

    def get_url(self, style_preset_id: str) -> str | None:
        path = self.get_path(style_preset_id)
        if not self._validate_path(path):
            return

        url = self._invoker.services.urls.get_style_preset_image_url(style_preset_id)

        # The image URL never changes, so we must add random query string to it to prevent caching
        url += f"?{uuid_string()}"

        return url

    def delete(self, style_preset_id: str) -> None:
        try:
            path = self.get_path(style_preset_id)

            if not self._validate_path(path):
                raise StylePresetImageFileNotFoundException

            path.unlink()

        except StylePresetImageFileNotFoundException as e:
            raise StylePresetImageFileNotFoundException from e
        except Exception as e:
            raise StylePresetImageFileDeleteException from e

    def _validate_path(self, path: Path) -> bool:
        """Validates the path given for an image."""
        return path.exists()

    def _validate_storage_folders(self) -> None:
        """Checks if the required folders exist and create them if they don't"""
        self._style_preset_images_folder.mkdir(parents=True, exist_ok=True)
