# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
import io
import json
import os
import shutil
import tempfile
import threading
import zlib
from collections.abc import Collection, Sequence
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
    """Files moved aside by ``stage_delete()``, restorable until the token is committed."""

    directory: Path
    files: list[tuple[Path, Path]]
    image_name: str
    image_subfolder: str


@dataclass
class _PendingDelete:
    """A durable record of an intent to purge files, written before the records are deleted.

    Nothing is moved: the journal directory names the images whose files are about to become
    unreferenced. Startup recovery reconciles any journal that outlives its operation by asking the
    record store which of its images are really gone.
    """

    directory: Path
    images: list[tuple[str, str]]


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
        # Guards the cache structures (__cache / __cache_ids), which are read and mutated from
        # multiple session-processor worker threads in multi-GPU parallel mode.
        self.__cache_lock = threading.Lock()

        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        # Validate required output folders at launch
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self.__recover_pending_deletes()

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
            # Image.open() is lazy: it reads the header but defers pixel decoding (and holds the
            # file handle open) until the first .load()/.copy()/.convert(). The opened object is
            # cached and the SAME object is handed to every caller, so in multi-GPU parallel mode
            # two worker threads can call .copy() on it concurrently and race on the shared file
            # handle and decoder state, producing "broken data stream" / "self.png is not None"
            # errors. Forcing the decode here makes the cached object safe for concurrent reads.
            image.load()
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
        image_path: Optional[Path] = None
        thumbnail_path: Optional[Path] = None
        image_existed = False
        thumbnail_existed = False
        try:
            self.__validate_storage_folders()
            image_path = self.get_path(image_name, image_subfolder=image_subfolder)
            image_existed = image_path.exists()

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

            thumbnail_path = self.get_path(image_name, thumbnail=True, image_subfolder=image_subfolder)
            thumbnail_existed = thumbnail_path.exists()

            # Build the thumbnail before replacing image.info with Invoke metadata. PIL stores
            # palette transparency there, and it must remain available to make_thumbnail().
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            thumbnail_image = make_thumbnail(image, thumbnail_size)

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

            thumbnail_image.save(thumbnail_path)

            self.__set_cache(image_path, image)
            self.__set_cache(thumbnail_path, thumbnail_image)
        except Exception as e:
            # A thumbnail failure must not leave a full-size image with no thumbnail. The
            # names are normally new, but preserve any pre-existing files when save() is
            # used to overwrite an existing image.
            for path, existed in ((image_path, image_existed), (thumbnail_path, thumbnail_existed)):
                if path is not None and not existed:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            self.evict_cache_paths([path for path in (image_path, thumbnail_path) if path is not None])
            raise ImageFileSaveException from e

    def delete(self, image_name: str, image_subfolder: str = "") -> None:
        token = self.stage_delete(image_name, image_subfolder)
        self.commit_delete(token)

    def stage_delete(self, image_name: str, image_subfolder: str = "") -> _StagedDelete:
        candidates = self.__delete_candidates(image_name, image_subfolder)
        staging_dir = Path(tempfile.mkdtemp(prefix=".delete_", dir=self.__output_folder))
        staged: list[tuple[Path, Path]] = []
        try:
            with open(staging_dir / "manifest.json", "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"image_name": image_name, "image_subfolder": image_subfolder}))
                manifest.flush()
                os.fsync(manifest.fileno())
            for index, source in enumerate(candidates):
                with self.__cache_lock:
                    self.__cache.pop(source, None)
                if source.exists():
                    destination = staging_dir / str(index)
                    source.replace(destination)
                    staged.append((source, destination))
            return _StagedDelete(
                directory=staging_dir, files=staged, image_name=image_name, image_subfolder=image_subfolder
            )
        except Exception as e:
            for source, destination in reversed(staged):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ImageFileDeleteException from e

    def begin_delete(self, images: Sequence[tuple[str, str]]) -> _PendingDelete:
        """Durably records the intent to purge these images' files, before their records are deleted.

        Callers delete the records first and purge afterwards, so the only inconsistency that can
        outlive a crash or a storage failure is a file nobody references. The journal written here
        is what makes that recoverable: ``__recover_pending_deletes()`` asks the record store about
        every image it names, purges the ones whose record is gone, and leaves the rest untouched.
        """
        # Resolve every path up front. A name that cannot be turned into a path must fail here,
        # while the caller can still abort — not after it has deleted the records.
        for image_name, image_subfolder in images:
            self.__delete_candidates(image_name, image_subfolder)
        journal_dir = Path(tempfile.mkdtemp(prefix=".delete_", dir=self.__output_folder))
        try:
            entries = [
                {"image_name": image_name, "image_subfolder": image_subfolder} for image_name, image_subfolder in images
            ]
            manifest_path = journal_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"version": 2, "images": entries}))
                manifest.flush()
                os.fsync(manifest.fileno())
            # Fsync the directory too: without it a crash can leave a journal directory whose
            # manifest entry never reached the disk, which recovery cannot act on.
            self.__fsync_directory(journal_dir)
            return _PendingDelete(directory=journal_dir, images=[(name, subfolder) for name, subfolder in images])
        except Exception as e:
            shutil.rmtree(journal_dir, ignore_errors=True)
            raise ImageFileDeleteException from e

    def abandon_delete(self, token: object) -> None:
        """Drops a pending-delete journal without purging anything.

        Used when the record deletion failed: the images are still live, so their files must stay.
        """
        if not isinstance(token, _PendingDelete):
            raise ImageFileDeleteException("Invalid pending-delete token")
        shutil.rmtree(token.directory, ignore_errors=True)

    def commit_delete(self, token: object, image_names: Optional[Collection[str]] = None) -> None:
        if isinstance(token, _PendingDelete):
            self.__commit_pending_delete(token, image_names)
            return
        if not isinstance(token, _StagedDelete):
            raise ImageFileDeleteException("Invalid staged-delete token")
        try:
            shutil.rmtree(token.directory)
        except Exception as e:
            raise ImageFileDeleteException from e

    def __commit_pending_delete(self, token: _PendingDelete, image_names: Optional[Collection[str]]) -> None:
        # ``image_names`` narrows the purge to the records that were actually deleted; the journal
        # still lists every candidate, which is harmless because recovery re-checks each one against
        # the record store and skips any that survived.
        selected = image_names if image_names is None else set(image_names)
        failures: list[str] = []
        for image_name, image_subfolder in token.images:
            if selected is not None and image_name not in selected:
                continue
            try:
                self.__purge_files(image_name, image_subfolder)
            except OSError as e:
                failures.append(f"{image_name}: {e}")
        if failures:
            # Leave the journal in place so the next startup retries every entry whose record is
            # gone. Removing it here would turn a transient storage error into a permanent orphan.
            raise ImageFileDeleteException(f"Failed to purge deleted image files: {'; '.join(failures)}")
        shutil.rmtree(token.directory, ignore_errors=True)

    def rollback_delete(self, token: object) -> None:
        if not isinstance(token, _StagedDelete):
            raise ImageFileDeleteException("Invalid staged-delete token")
        try:
            for source, destination in reversed(token.files):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            # While these files sat in the staging directory another request may have deleted the
            # record; restoring them would leave files nothing references and no journal to find
            # them by. Re-check now that the files are back: every deleter purges an image's files
            # only *after* its record is committed as gone, so a record still present here cannot
            # have been purged before this restore, and a record already absent means the purge
            # either found nothing or is still to come — either way the files must go.
            self.__purge_if_record_absent(token.image_name, token.image_subfolder)
            shutil.rmtree(token.directory, ignore_errors=True)
        except Exception as e:
            raise ImageFileDeleteException from e

    def __delete_candidates(self, image_name: str, image_subfolder: str) -> list[Path]:
        return [
            self.get_path(image_name, image_subfolder=image_subfolder),
            self.get_path(image_name, thumbnail=True, image_subfolder=image_subfolder),
        ]

    def __purge_files(self, image_name: str, image_subfolder: str) -> None:
        """Removes an image's file and thumbnail. Missing files are not an error."""
        for path in self.__delete_candidates(image_name, image_subfolder):
            with self.__cache_lock:
                self.__cache.pop(path, None)
            path.unlink(missing_ok=True)

    def __purge_if_record_absent(self, image_name: str, image_subfolder: str) -> None:
        try:
            record_exists = self.__invoker.services.image_records.exists(image_name)
        except Exception as e:
            # A storage fault must never destroy a live image's files. Keep them: a stale file is
            # recoverable at the next startup, a deleted one is not.
            InvokeAILogger.get_logger().error(f"Could not confirm whether {image_name} still exists: {e}")
            return
        if record_exists:
            return
        self.__purge_files(image_name, image_subfolder)

    @staticmethod
    def __fsync_directory(directory: Path) -> None:
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            # Windows cannot open a directory for fsync; the manifest write above is all we get.
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

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

    def __recover_pending_deletes(self) -> None:
        """Reconciles every delete journal left behind by an interrupted or failed deletion.

        One rule covers both journal shapes, and the record store decides it: an image whose record
        survives was never really deleted, so anything staged for it is put back and the journal
        dropped; an image whose record is gone is an orphan, so its files are purged wherever the
        interrupted operation left them.
        """
        logger = InvokeAILogger.get_logger()
        for journal_dir in sorted(self.__output_folder.glob(".delete_*")):
            manifest_path = journal_dir / "manifest.json"
            if not manifest_path.is_file():
                # mkdtemp() ran but the manifest never landed, so this directory names nothing and
                # there is nothing to reconcile. Only remove it when it is empty.
                if not any(journal_dir.iterdir()):
                    journal_dir.rmdir()
                continue
            try:
                with open(manifest_path, encoding="utf-8") as manifest:
                    data = json.load(manifest)
                for image_name, image_subfolder in self.__manifest_images(data):
                    if self.__invoker.services.image_records.exists(image_name):
                        # Put back whatever stage_delete() moved aside. Only a single-image journal
                        # ever holds staged files, at indices 0 and 1; a pending-delete journal
                        # moves nothing, so these lookups simply find nothing to restore.
                        for index, source in enumerate(self.__delete_candidates(image_name, image_subfolder)):
                            staged = journal_dir / str(index)
                            if staged.exists():
                                source.parent.mkdir(parents=True, exist_ok=True)
                                staged.replace(source)
                        continue
                    self.__purge_files(image_name, image_subfolder)
                shutil.rmtree(journal_dir, ignore_errors=True)
            except Exception as error:
                # Includes a record-store fault: leave the journal for the next startup rather than
                # guess. Retrying is always safe; both branches above are idempotent.
                logger.error(f"Failed to recover image deletion journal {journal_dir}: {error}")

    @staticmethod
    def __manifest_images(data: dict) -> list[tuple[str, str]]:
        """Reads both journal shapes: a pending delete lists many images, a staged delete names one."""
        entries = data.get("images")
        if entries is None:
            return [(data["image_name"], data.get("image_subfolder", ""))]
        return [(entry["image_name"], entry.get("image_subfolder", "")) for entry in entries]

    def __get_cache(self, image_name: Path) -> Optional[PILImageType]:
        with self.__cache_lock:
            return None if image_name not in self.__cache else self.__cache[image_name]

    def __set_cache(self, image_name: Path, image: PILImageType):
        with self.__cache_lock:
            if image_name not in self.__cache:
                self.__cache[image_name] = image
                self.__cache_ids.put(image_name)  # TODO: this should refresh position for LRU cache
                if len(self.__cache) > self.__max_cache_size:
                    cache_id = self.__cache_ids.get()
                    if cache_id in self.__cache:
                        del self.__cache[cache_id]
