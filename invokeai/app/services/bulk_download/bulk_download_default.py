import os
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union
from zipfile import ZipFile

from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.bulk_download.bulk_download_base import BulkDownloadBase
from invokeai.app.services.bulk_download.bulk_download_common import (
    DEFAULT_BULK_DOWNLOAD_ID,
    BulkDownloadBusyError,
    BulkDownloadException,
    BulkDownloadParametersException,
    BulkDownloadStatus,
    BulkDownloadTargetException,
)
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException
from invokeai.app.services.images.images_common import ImageDTO, ImageExportFormat, ImageExportUnavailableError
from invokeai.app.services.invoker import Invoker
from invokeai.app.util.misc import uuid_string
from invokeai.backend.image_util.raster.descriptor import RasterConversionError
from invokeai.backend.image_util.raster.png_codec import PngCodecError
from invokeai.backend.image_util.raster.precise_asset import PreciseAssetError

# A finished archive that nobody fetched is deleted when a later job starts, once it is this old.
_UNFETCHED_ARCHIVE_MAX_AGE_SECONDS = 3600

# Safety bound on how long `stop` waits for a running job. The normal wait is the image being encoded, because the job
# removes its part file itself and Windows cannot delete the temp directory while that file is open.
_STOP_JOIN_SECONDS = 600.0

# Image failures that leave the gallery file in the archive, with one warning. A missing record is not among them.
_PNG16_FALLBACK_ERRORS = (
    RasterConversionError,
    PreciseAssetError,
    PngCodecError,
    ImageFileNotFoundException,
    OSError,
    MemoryError,
)


class BulkDownloadService(BulkDownloadBase):
    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def __init__(self):
        self._temp_directory = TemporaryDirectory()
        self._bulk_downloads_folder = Path(self._temp_directory.name) / "bulk_downloads"
        self._bulk_downloads_folder.mkdir(parents=True, exist_ok=True)
        # Track which user owns each download so the fetch endpoint can enforce ownership
        self._download_owners: dict[str, str] = {}
        # Item name -> None while the archive is being built, or the reason it failed (until reported once).
        self._jobs: dict[str, Optional[str]] = {}
        # The item holding the single 16-bit job. Its encodes run outside the image service's export slot, so this
        # bounds them to one image in memory next to at most one single export.
        self._png16_job: Optional[str] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: set[threading.Thread] = set()

    def start_job(
        self,
        image_names: Optional[list[str]],
        board_id: Optional[str],
        bulk_download_item_id: str,
        user_id: str,
        export_format: Optional[ImageExportFormat] = None,
        uncategorized_owner_id: Optional[str] = None,
    ) -> None:
        # Not a Starlette background task: the server waits for those before it shuts the services down, so `stop`
        # would only run after a long archive had finished.
        self._register(bulk_download_item_id, user_id, export_format)
        thread = threading.Thread(
            target=self._run_job,
            args=(image_names, board_id, bulk_download_item_id, user_id, export_format, uncategorized_owner_id),
            name=f"bulk-download-{bulk_download_item_id}",
            daemon=True,
        )
        with self._lock:
            self._threads.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._lock:
                self._threads.discard(thread)
                self._jobs.pop(bulk_download_item_id + ".zip", None)
                self._download_owners.pop(bulk_download_item_id + ".zip", None)
                if self._png16_job == bulk_download_item_id + ".zip":
                    self._png16_job = None
            raise

    def _run_job(self, *args) -> None:
        try:
            self.handler(*args)
        except Exception:
            # The handler has recorded and signalled the failure; the thread has nobody to re-raise to.
            self._invoker.services.logger.exception("Bulk download job failed")
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _register(self, bulk_download_item_id: str, user_id: str, export_format: Optional[ImageExportFormat]) -> None:
        item_name = bulk_download_item_id + ".zip"
        with self._lock:
            if export_format is ImageExportFormat.PNG16:
                if self._png16_job is not None:
                    raise BulkDownloadBusyError()
                self._png16_job = item_name
            # Record ownership so the fetch endpoint can verify the caller
            self._download_owners[item_name] = user_id
            self._jobs[item_name] = None

    def get_status(self, bulk_download_item_name: str) -> BulkDownloadStatus:
        try:
            self.get_path(bulk_download_item_name)
            return BulkDownloadStatus.READY
        except BulkDownloadTargetException:
            pass
        with self._lock:
            if bulk_download_item_name not in self._jobs:
                return BulkDownloadStatus.UNKNOWN
            if self._jobs[bulk_download_item_name] is None:
                return BulkDownloadStatus.BUILDING
            return BulkDownloadStatus.FAILED

    def pop_failure(self, bulk_download_item_name: str) -> Optional[str]:
        with self._lock:
            reason = self._jobs.pop(bulk_download_item_name, None)
            if reason is not None:
                self._download_owners.pop(bulk_download_item_name, None)
            return reason

    def handler(
        self,
        image_names: Optional[list[str]],
        board_id: Optional[str],
        bulk_download_item_id: Optional[str],
        user_id: str = "system",
        export_format: Optional[ImageExportFormat] = None,
        uncategorized_owner_id: Optional[str] = None,
    ) -> None:
        bulk_download_id: str = DEFAULT_BULK_DOWNLOAD_ID
        bulk_download_item_id = bulk_download_item_id or uuid_string()
        bulk_download_item_name = bulk_download_item_id + ".zip"

        with self._lock:
            registered = bulk_download_item_name in self._jobs
        if not registered:
            self._register(bulk_download_item_id, user_id, export_format)

        try:
            self._signal_job_started(bulk_download_id, bulk_download_item_id, bulk_download_item_name, user_id)
            self._delete_unfetched_archives()
            image_dtos: list[ImageDTO] = []

            if board_id:
                image_dtos = self._board_handler(board_id, uncategorized_owner_id)
            elif image_names:
                image_dtos = self._image_handler(image_names)
            else:
                raise BulkDownloadParametersException()

            bulk_download_item_name: str = self._create_zip_file(image_dtos, bulk_download_item_id, export_format)
            with self._lock:
                self._jobs.pop(bulk_download_item_name, None)
            self._signal_job_completed(bulk_download_id, bulk_download_item_id, bulk_download_item_name, user_id)
        except (
            ImageRecordNotFoundException,
            BoardRecordNotFoundException,
            BulkDownloadException,
            BulkDownloadParametersException,
        ) as e:
            self._record_failure(bulk_download_item_name, e)
            self._signal_job_failed(bulk_download_id, bulk_download_item_id, bulk_download_item_name, e, user_id)
        except Exception as e:
            self._record_failure(bulk_download_item_name, e)
            self._signal_job_failed(bulk_download_id, bulk_download_item_id, bulk_download_item_name, e, user_id)
            self._invoker.services.logger.error("Problem bulk downloading images.")
            raise e
        finally:
            with self._lock:
                if self._png16_job == bulk_download_item_name:
                    self._png16_job = None

    def _record_failure(self, bulk_download_item_name: str, error: Exception) -> None:
        with self._lock:
            self._jobs[bulk_download_item_name] = str(error) or type(error).__name__

    def _delete_unfetched_archives(self) -> None:
        cutoff = time.time() - _UNFETCHED_ARCHIVE_MAX_AGE_SECONDS
        for archive in self._bulk_downloads_folder.glob("*.zip"):
            try:
                with self._lock:
                    if archive.name in self._jobs or archive.stat().st_mtime >= cutoff:
                        continue
                    self._download_owners.pop(archive.name, None)
                archive.unlink()
            except OSError:
                continue

    def _image_handler(self, image_names: list[str]) -> list[ImageDTO]:
        return [self._invoker.services.images.get_dto(image_name) for image_name in image_names]

    def _board_handler(self, board_id: str, uncategorized_owner_id: Optional[str]) -> list[ImageDTO]:
        # "none" spans every account's uncategorized images, so it is scoped to the owner. A real board was
        # authorized as a whole, so a shared board keeps every contributor's images.
        image_names = self._invoker.services.board_image_records.get_all_board_image_names_for_board(
            board_id,
            categories=None,
            is_intermediate=None,
            user_id=uncategorized_owner_id if board_id == "none" else None,
        )
        return self._image_handler(image_names)

    def generate_item_id(self, board_id: Optional[str]) -> str:
        return uuid_string() if board_id is None else self._get_clean_board_name(board_id) + "_" + uuid_string()

    def _get_clean_board_name(self, board_id: str) -> str:
        if board_id == "none":
            return "Uncategorized"

        return self._clean_string_to_path_safe(self._invoker.services.board_records.get(board_id).board_name)

    def _create_zip_file(
        self, image_dtos: list[ImageDTO], bulk_download_item_id: str, export_format: Optional[ImageExportFormat] = None
    ) -> str:
        """
        Create a zip file containing the images specified by the given image names or board id.
        If download with the same bulk_download_id already exists, it will be overwritten.

        :return: The name of the zip file.
        """
        zip_file_name = bulk_download_item_id + ".zip"
        zip_file_path = self._bulk_downloads_folder / (zip_file_name)
        # The fetch endpoint serves `<id>.zip` as soon as it exists, so the archive is built beside it and moved
        # into place once complete.
        part_path = zip_file_path.with_name(zip_file_name + ".part")

        try:
            with ZipFile(part_path, "w") as zip_file:
                for image_dto in image_dtos:
                    if self._stop_event.is_set():
                        raise BulkDownloadException("The bulk download was stopped")
                    image_zip_path = Path(image_dto.image_category.value) / image_dto.image_name
                    if export_format is ImageExportFormat.PNG16 and self._write_png16(zip_file, image_dto):
                        continue
                    image_disk_path = self._invoker.services.images.get_path(image_dto.image_name)
                    zip_file.write(image_disk_path, arcname=image_zip_path)
            os.replace(part_path, zip_file_path)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise

        return str(zip_file_name)

    def _write_png16(self, zip_file: ZipFile, image_dto: ImageDTO) -> bool:
        """Stores the image's 16-bit PNG in the zip. Returns False when the gallery file is to be stored instead:
        silently for an image without an exportable precise asset, with one warning when the encode fails.

        Only this job encodes, so one image's bytes are in memory at a time; they are dropped before the next."""
        image_name = image_dto.image_name
        try:
            content = self._invoker.services.images.encode_png16(image_name)
        except ImageExportUnavailableError:
            return False
        except _PNG16_FALLBACK_ERRORS as e:
            self._invoker.services.logger.warning(
                f"Bulk download: the 16-bit PNG of {image_name} could not be made, so the archive holds its gallery "
                f"file: {e}"
            )
            return False
        arcname = Path(image_dto.image_category.value) / f"{Path(image_name).stem}-16bit.png"
        zip_file.writestr(str(arcname), content)
        return True

    # from https://stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string
    def _clean_string_to_path_safe(self, s: str) -> str:
        """Clean a string to be path safe."""
        return "".join([c for c in s if c.isalpha() or c.isdigit() or c == " " or c == "_" or c == "-"]).rstrip()

    def _signal_job_started(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        user_id: str = "system",
    ) -> None:
        """Signal that a bulk download job has started."""
        if self._invoker:
            assert bulk_download_id is not None
            self._invoker.services.events.emit_bulk_download_started(
                bulk_download_id, bulk_download_item_id, bulk_download_item_name, user_id=user_id
            )

    def _signal_job_completed(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        user_id: str = "system",
    ) -> None:
        """Signal that a bulk download job has completed."""
        if self._invoker:
            assert bulk_download_id is not None
            assert bulk_download_item_name is not None
            self._invoker.services.events.emit_bulk_download_complete(
                bulk_download_id, bulk_download_item_id, bulk_download_item_name, user_id=user_id
            )

    def _signal_job_failed(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        exception: Exception,
        user_id: str = "system",
    ) -> None:
        """Signal that a bulk download job has failed."""
        if self._invoker:
            assert bulk_download_id is not None
            assert exception is not None
            self._invoker.services.events.emit_bulk_download_error(
                bulk_download_id, bulk_download_item_id, bulk_download_item_name, str(exception), user_id=user_id
            )

    def stop(self, *args, **kwargs):
        self._stop_event.set()
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            if thread is threading.current_thread():
                continue
            # The job ends after the image it is encoding and removes its part file; the directory cannot be
            # deleted before that on Windows.
            thread.join(timeout=_STOP_JOIN_SECONDS)
            if thread.is_alive():
                self._invoker.services.logger.warning(f"Bulk download job {thread.name} did not end at shutdown")
        try:
            self._temp_directory.cleanup()
        except OSError as e:
            self._invoker.services.logger.warning(f"Could not remove the bulk download directory: {e}")

    def get_owner(self, bulk_download_item_name: str) -> Optional[str]:
        return self._download_owners.get(bulk_download_item_name)

    def delete(self, bulk_download_item_name: str) -> None:
        path = self.get_path(bulk_download_item_name)
        Path(path).unlink()
        self._download_owners.pop(bulk_download_item_name, None)

    def get_path(self, bulk_download_item_name: str) -> str:
        path = str(self._bulk_downloads_folder / bulk_download_item_name)
        if not self._is_valid_path(path):
            raise BulkDownloadTargetException()
        return path

    def _is_valid_path(self, path: Union[str, Path]) -> bool:
        """Validates the path given for a bulk download."""
        path = path if isinstance(path, Path) else Path(path)

        # Resolve the path to handle any path traversal attempts (e.g., ../)
        resolved_path = path.resolve()

        # The path may not traverse out of the bulk downloads folder or its subfolders
        does_not_traverse = resolved_path.parent == self._bulk_downloads_folder.resolve()

        # The path must exist and be a .zip file
        does_exist = resolved_path.exists()
        is_zip_file = resolved_path.suffix == ".zip"

        return does_exist and is_zip_file and does_not_traverse
