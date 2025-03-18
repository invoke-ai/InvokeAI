from pathlib import Path

from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowCategory
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_base import WorkflowThumbnailServiceBase
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_common import (
    WorkflowThumbnailFileDeleteException,
    WorkflowThumbnailFileNotFoundException,
    WorkflowThumbnailFileSaveException,
)
from invokeai.app.util.misc import uuid_string
from invokeai.app.util.thumbnails import make_thumbnail


class WorkflowThumbnailFileStorageDisk(WorkflowThumbnailServiceBase):
    def __init__(self, thumbnails_path: Path):
        self._workflow_thumbnail_folder = thumbnails_path
        self._validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, workflow_id: str) -> PILImageType:
        try:
            path = self.get_path(workflow_id)

            return Image.open(path)
        except FileNotFoundError as e:
            raise WorkflowThumbnailFileNotFoundException from e

    def save(self, workflow_id: str, image: PILImageType) -> None:
        try:
            self._validate_storage_folders()
            image_path = self._workflow_thumbnail_folder / (workflow_id + ".webp")
            thumbnail = make_thumbnail(image, 256)
            thumbnail.save(image_path, format="webp")

        except Exception as e:
            raise WorkflowThumbnailFileSaveException from e

    def get_path(self, workflow_id: str, with_hash: bool = True) -> Path:
        workflow = self._invoker.services.workflow_records.get(workflow_id).workflow
        if workflow.meta.category is WorkflowCategory.Default:
            default_thumbnails_dir = Path(__file__).parent / Path("default_workflow_thumbnails")
            path = default_thumbnails_dir / (workflow_id + ".png")
        else:
            path = self._workflow_thumbnail_folder / (workflow_id + ".webp")

        return path

    def get_url(self, workflow_id: str, with_hash: bool = True) -> str | None:
        path = self.get_path(workflow_id)
        if not self._validate_path(path):
            return

        url = self._invoker.services.urls.get_workflow_thumbnail_url(workflow_id)

        # The image URL never changes, so we must add random query string to it to prevent caching
        if with_hash:
            url += f"?{uuid_string()}"

        return url

    def delete(self, workflow_id: str) -> None:
        try:
            path = self.get_path(workflow_id)

            if not self._validate_path(path):
                raise WorkflowThumbnailFileNotFoundException

            path.unlink()

        except WorkflowThumbnailFileNotFoundException as e:
            raise WorkflowThumbnailFileNotFoundException from e
        except Exception as e:
            raise WorkflowThumbnailFileDeleteException from e

    def _validate_path(self, path: Path) -> bool:
        """Validates the path given for an image."""
        return path.exists()

    def _validate_storage_folders(self) -> None:
        """Checks if the required folders exist and create them if they don't"""
        self._workflow_thumbnail_folder.mkdir(parents=True, exist_ok=True)
