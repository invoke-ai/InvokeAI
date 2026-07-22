# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
import io
import json
import os
import shutil
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Optional, Union

from PIL import Image, PngImagePlugin
from PIL.Image import Image as PILImageType

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.util.thumbnails import get_thumbnail_name, make_thumbnail
from invokeai.backend.util.logging import InvokeAILogger

_PNG_RLE_MIN_PIXELS = 512 * 512
_PNG_RLE_SAMPLE_TILE = 32
_PNG_RLE_MIN_RAW_SIZE_PERCENT = 30
_PNG_RLE_MAX_SAMPLE_SIZE_PERCENT = 102


@dataclass
class _StagedDelete:
    directory: Path
    files: list[tuple[Path, Path]]


def _get_png_size(image: PILImageType, compress_type: Optional[int] = None) -> int:
    output = io.BytesIO()
    options = {"compress_level": 1}
    if compress_type is not None:
        options["compress_type"] = compress_type
    image.save(output, "PNG", **options)
    return output.tell()


def _should_use_png_rle(image: PILImageType) -> bool:
    if image.mode not in {"RGB", "RGBA"} or image.width * image.height < _PNG_RLE_MIN_PIXELS:
        return False

    # Native-resolution tiles distinguish high-entropy data from filter-friendly structured images.
    tile_width = min(_PNG_RLE_SAMPLE_TILE, image.width)
    tile_height = min(_PNG_RLE_SAMPLE_TILE, image.height)
    x_positions = (0, (image.width - tile_width) // 2, image.width - tile_width)
    y_positions = (0, (image.height - tile_height) // 2, image.height - tile_height)
    sample = Image.new(image.mode, (tile_width * 3, tile_height * 3))
    for row, y in enumerate(y_positions):
        for column, x in enumerate(x_positions):
            with image.crop((x, y, x + tile_width, y + tile_height)) as tile:
                sample.paste(tile, (column * tile_width, row * tile_height))

    try:
        raw = sample.tobytes()
        if len(zlib.compress(raw, level=1)) * 100 < len(raw) * _PNG_RLE_MIN_RAW_SIZE_PERCENT:
            return False
        default_size = _get_png_size(sample)
        rle_size = _get_png_size(sample, zlib.Z_RLE)
        return rle_size * 100 <= default_size * _PNG_RLE_MAX_SAMPLE_SIZE_PERCENT
    finally:
        sample.close()


class DiskImageFileStorage(ImageFileStorageBase):
    """Stores images on disk"""

    def __init__(self, output_folder: Union[str, Path]):
        self.__cache: dict[Path, PILImageType] = {}
        self.__cache_ids = Queue[Path]()
        self.__max_cache_size = 10  # TODO: get this from config

        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        # Validate required output folders at launch
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self.__recover_staged_deletes()

    @property
    def image_root(self) -> Path:
        return self.__output_folder.resolve()

    @property
    def thumbnail_root(self) -> Path:
        return self.__thumbnails_folder.resolve()

    def evict_cache_paths(self, paths: list[Path]) -> None:
        for path in paths:
            self.__cache.pop(path.resolve(), None)

    def get(self, image_name: str, image_subfolder: str = "") -> PILImageType:
        try:
            image_path = self.get_path(image_name, image_subfolder=image_subfolder)

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
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        thumbnail_size: int = 256,
        image_subfolder: str = "",
    ) -> None:
        try:
            self.__validate_storage_folders()
            image_path = self.get_path(image_name, image_subfolder=image_subfolder)

            # Ensure subfolder directories exist
            image_path.parent.mkdir(parents=True, exist_ok=True)

            pnginfo = PngImagePlugin.PngInfo()
            info_dict = {}

            if metadata is not None:
                info_dict["invokeai_metadata"] = metadata
                pnginfo.add_text("invokeai_metadata", metadata)
            if workflow is not None:
                info_dict["invokeai_workflow"] = workflow
                pnginfo.add_text("invokeai_workflow", workflow)
            if graph is not None:
                info_dict["invokeai_graph"] = graph
                pnginfo.add_text("invokeai_graph", graph)

            # When saving the image, the image object's info field is not populated. We need to set it
            image.info = info_dict
            compress_level = self.__invoker.services.configuration.pil_compress_level
            save_options = {"compress_level": compress_level}
            if compress_level == 1 and _should_use_png_rle(image):
                save_options["compress_type"] = zlib.Z_RLE
            image.save(
                image_path,
                "PNG",
                pnginfo=pnginfo,
                **save_options,
            )

            thumbnail_path = self.get_path(image_name, thumbnail=True, image_subfolder=image_subfolder)

            # Ensure thumbnail subfolder directories exist
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

            thumbnail_image = make_thumbnail(image, thumbnail_size)
            thumbnail_image.save(thumbnail_path)

            self.__set_cache(image_path, image)
            self.__set_cache(thumbnail_path, thumbnail_image)
        except Exception as e:
            raise ImageFileSaveException from e

    def delete(self, image_name: str, image_subfolder: str = "") -> None:
        token = self.stage_delete(image_name, image_subfolder)
        self.commit_delete(token)

    def stage_delete(self, image_name: str, image_subfolder: str = "") -> _StagedDelete:
        candidates = [
            self.get_path(image_name, image_subfolder=image_subfolder),
            self.get_path(image_name, thumbnail=True, image_subfolder=image_subfolder),
        ]
        staging_dir = Path(tempfile.mkdtemp(prefix=".delete_", dir=self.__output_folder))
        staged: list[tuple[Path, Path]] = []
        try:
            with open(staging_dir / "manifest.json", "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"image_name": image_name, "image_subfolder": image_subfolder}))
                manifest.flush()
                os.fsync(manifest.fileno())
            for index, source in enumerate(candidates):
                self.__cache.pop(source, None)
                if source.exists():
                    destination = staging_dir / str(index)
                    source.replace(destination)
                    staged.append((source, destination))
            return _StagedDelete(directory=staging_dir, files=staged)
        except Exception as e:
            for source, destination in reversed(staged):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ImageFileDeleteException from e

    def commit_delete(self, token: object) -> None:
        if not isinstance(token, _StagedDelete):
            raise ImageFileDeleteException("Invalid staged-delete token")
        try:
            shutil.rmtree(token.directory)
        except Exception as e:
            raise ImageFileDeleteException from e

    def rollback_delete(self, token: object) -> None:
        if not isinstance(token, _StagedDelete):
            raise ImageFileDeleteException("Invalid staged-delete token")
        try:
            for source, destination in reversed(token.files):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            shutil.rmtree(token.directory, ignore_errors=True)
        except Exception as e:
            raise ImageFileDeleteException from e

    def get_path(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> Path:
        base_folder = self.__thumbnails_folder if thumbnail else self.__output_folder
        filename = get_thumbnail_name(image_name) if thumbnail else image_name

        # Validate the filename itself (no path separators allowed in the filename)
        basename = Path(filename).name
        if basename != filename:
            raise ValueError("Invalid image name, potential directory traversal detected")

        # Build the full path with optional subfolder
        if image_subfolder:
            self._validate_subfolder(image_subfolder)
            image_path = base_folder / image_subfolder / basename
        else:
            image_path = base_folder / basename

        # Ensure the image path is within the base folder to prevent directory traversal
        resolved_base = base_folder.resolve()
        resolved_image_path = image_path.resolve()

        if not resolved_image_path.is_relative_to(resolved_base):
            raise ValueError("Image path outside outputs folder, potential directory traversal detected")

        return resolved_image_path

    @staticmethod
    def _validate_subfolder(subfolder: str) -> None:
        """Validates a subfolder path to prevent directory traversal while allowing controlled subdirectories."""
        if not subfolder:
            return
        if "\\" in subfolder:
            raise ValueError("Backslashes not allowed in subfolder path")
        if subfolder.startswith("/"):
            raise ValueError("Absolute paths not allowed in subfolder path")
        parts = subfolder.split("/")
        for part in parts:
            if part == "..":
                raise ValueError("Parent directory references not allowed in subfolder path")
            if part == "":
                raise ValueError("Empty path segments not allowed in subfolder path")

    def validate_path(self, path: Union[str, Path]) -> bool:
        """Validates the path given for an image or thumbnail."""
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    def get_workflow(self, image_name: str, image_subfolder: str = "") -> str | None:
        image = self.get(image_name, image_subfolder=image_subfolder)
        workflow = image.info.get("invokeai_workflow", None)
        if isinstance(workflow, str):
            return workflow
        return None

    def get_graph(self, image_name: str, image_subfolder: str = "") -> str | None:
        image = self.get(image_name, image_subfolder=image_subfolder)
        graph = image.info.get("invokeai_graph", None)
        if isinstance(graph, str):
            return graph
        return None

    def __validate_storage_folders(self) -> None:
        """Checks if the required output folders exist and create them if they don't"""
        folders: list[Path] = [self.__output_folder, self.__thumbnails_folder]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    def __recover_staged_deletes(self) -> None:
        logger = InvokeAILogger.get_logger()
        for staging_dir in self.__output_folder.glob(".delete_*"):
            manifest_path = staging_dir / "manifest.json"
            if not manifest_path.is_file():
                if not any(staging_dir.iterdir()):
                    staging_dir.rmdir()
                continue
            try:
                with open(manifest_path, encoding="utf-8") as manifest:
                    data = json.load(manifest)
                image_name = data["image_name"]
                image_subfolder = data.get("image_subfolder", "")
                candidates = [
                    self.get_path(image_name, image_subfolder=image_subfolder),
                    self.get_path(image_name, thumbnail=True, image_subfolder=image_subfolder),
                ]
                token = _StagedDelete(
                    directory=staging_dir,
                    files=[(source, staging_dir / str(index)) for index, source in enumerate(candidates)],
                )
                self.__invoker.services.image_records.get(image_name)
                self.rollback_delete(token)
            except Exception as error:
                from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException

                if isinstance(error, ImageRecordNotFoundException):
                    shutil.rmtree(staging_dir, ignore_errors=True)
                else:
                    logger.error(f"Failed to recover staged image deletion {staging_dir}: {error}")

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
