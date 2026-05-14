from pathlib import Path
from typing import Optional, Union

from invokeai.app.services.canvas_project_files.canvas_project_files_base import CanvasProjectFileStorageBase
from invokeai.app.services.canvas_project_files.canvas_project_files_common import (
    CanvasProjectFileDeleteException,
    CanvasProjectFileSaveException,
)
from invokeai.app.services.invoker import Invoker
from invokeai.backend.util.logging import InvokeAILogger


def get_canvas_project_thumbnail_name(project_name: str) -> str:
    """Returns the thumbnail file name for a given project name (project name + .webp)."""
    return f"{project_name}.webp"


def get_canvas_project_file_name(project_name: str) -> str:
    """Returns the on-disk file name for a given project name (project name + .invk)."""
    return f"{project_name}.invk"


class DiskCanvasProjectFileStorage(CanvasProjectFileStorageBase):
    """Stores canvas project ZIP (.invk) files on disk under {outputs}/canvas_projects/, with optional
    WebP thumbnails under {outputs}/canvas_projects/thumbnails/."""

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def save(
        self,
        zip_bytes: bytes,
        project_name: str,
        thumbnail_bytes: Optional[bytes] = None,
        project_subfolder: str = "",
    ) -> None:
        logger = InvokeAILogger.get_logger()
        try:
            self.__validate_storage_folders()
            project_path = self.get_path(project_name, project_subfolder=project_subfolder)
            project_path.parent.mkdir(parents=True, exist_ok=True)
            with open(project_path, "wb") as f:
                f.write(zip_bytes)
            logger.info(f"Canvas project file written: {project_path}")

            if thumbnail_bytes is not None:
                thumbnail_path = self.get_path(project_name, thumbnail=True, project_subfolder=project_subfolder)
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                with open(thumbnail_path, "wb") as f:
                    f.write(thumbnail_bytes)
                logger.info(f"Canvas project thumbnail written: {thumbnail_path}")
        except Exception as e:
            raise CanvasProjectFileSaveException from e

    def delete(self, project_name: str, project_subfolder: str = "") -> None:
        try:
            project_path = self.get_path(project_name, project_subfolder=project_subfolder)
            if project_path.exists():
                project_path.unlink()

            thumbnail_path = self.get_path(project_name, thumbnail=True, project_subfolder=project_subfolder)
            if thumbnail_path.exists():
                thumbnail_path.unlink()
        except Exception as e:
            raise CanvasProjectFileDeleteException from e

    def get_path(self, project_name: str, thumbnail: bool = False, project_subfolder: str = "") -> Path:
        base_folder = self.__thumbnails_folder if thumbnail else self.__output_folder
        filename = (
            get_canvas_project_thumbnail_name(project_name) if thumbnail else get_canvas_project_file_name(project_name)
        )

        basename = Path(filename).name
        if basename != filename:
            raise ValueError("Invalid project name, potential directory traversal detected")

        if project_subfolder:
            self._validate_subfolder(project_subfolder)
            project_path = base_folder / project_subfolder / basename
        else:
            project_path = base_folder / basename

        resolved_base = base_folder.resolve()
        resolved_project_path = project_path.resolve()
        if not resolved_project_path.is_relative_to(resolved_base):
            raise ValueError("Project path outside outputs folder, potential directory traversal detected")
        return resolved_project_path

    def validate_path(self, path: Union[str, Path]) -> bool:
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    @staticmethod
    def _validate_subfolder(subfolder: str) -> None:
        """Validates a subfolder path to prevent directory traversal."""
        if not subfolder:
            return
        if "\\" in subfolder:
            raise ValueError("Backslashes not allowed in subfolder path")
        if subfolder.startswith("/"):
            raise ValueError("Absolute paths not allowed in subfolder path")
        for part in subfolder.split("/"):
            if part == "..":
                raise ValueError("Parent directory references not allowed in subfolder path")
            if part == "":
                raise ValueError("Empty path segments not allowed in subfolder path")

    def __validate_storage_folders(self) -> None:
        for folder in (self.__output_folder, self.__thumbnails_folder):
            folder.mkdir(parents=True, exist_ok=True)
