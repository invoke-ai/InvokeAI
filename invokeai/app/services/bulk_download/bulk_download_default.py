from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union
from zipfile import ZipFile

from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.bulk_download.bulk_download_common import (
    DEFAULT_BULK_DOWNLOAD_ID,
    BulkDownloadException,
    BulkDownloadParametersException,
    BulkDownloadTargetException,
)
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.util.misc import uuid_string

from .bulk_download_base import BulkDownloadBase


class BulkDownloadService(BulkDownloadBase):
    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._event_bus = invoker.services.events

    def __init__(self):
        self._temp_directory = TemporaryDirectory()
        self._bulk_downloads_folder = Path(self._temp_directory.name) / "bulk_downloads"
        self._bulk_downloads_folder.mkdir(parents=True, exist_ok=True)

    def handler(
        self, image_names: Optional[list[str]], board_id: Optional[str], bulk_download_item_id: Optional[str]
    ) -> None:
        bulk_download_id: str = DEFAULT_BULK_DOWNLOAD_ID
        bulk_download_item_id = uuid_string() if bulk_download_item_id is None else bulk_download_item_id
        bulk_download_item_name = bulk_download_item_id + ".zip"

        self._signal_job_started(bulk_download_id, bulk_download_item_id, bulk_download_item_name)

        try:
            image_dtos: list[ImageDTO] = []

            if board_id:
                image_dtos = self._board_handler(board_id)
            elif image_names:
                image_dtos = self._image_handler(image_names)
            else:
                raise BulkDownloadParametersException()

            bulk_download_item_name: str = self._create_zip_file(image_dtos, bulk_download_item_id)
            self._signal_job_completed(bulk_download_id, bulk_download_item_id, bulk_download_item_name)
        except (
            ImageRecordNotFoundException,
            BoardRecordNotFoundException,
            BulkDownloadException,
            BulkDownloadParametersException,
        ) as e:
            self._signal_job_failed(bulk_download_id, bulk_download_item_id, bulk_download_item_name, e)
        except Exception as e:
            self._signal_job_failed(bulk_download_id, bulk_download_item_id, bulk_download_item_name, e)
            self._invoker.services.logger.error("Problem bulk downloading images.")
            raise e

    def _image_handler(self, image_names: list[str]) -> list[ImageDTO]:
        return [self._invoker.services.images.get_dto(image_name) for image_name in image_names]

    def _board_handler(self, board_id: str) -> list[ImageDTO]:
        # -1 is the default value for limit, which means no limit, is_intermediate False only gives us completed images
        image_dtos = self._invoker.services.images.get_many(
            offset=0,
            limit=-1,
            board_id=board_id,
            is_intermediate=False,
        ).items
        return image_dtos

    def generate_item_id(self, board_id: Optional[str]) -> str:
        return uuid_string() if board_id is None else self._get_clean_board_name(board_id) + "_" + uuid_string()

    def _get_clean_board_name(self, board_id: str) -> str:
        if board_id == "none":
            return "Uncategorized"

        return self._clean_string_to_path_safe(self._invoker.services.board_records.get(board_id).board_name)

    def _create_zip_file(self, image_dtos: list[ImageDTO], bulk_download_item_id: str) -> str:
        """
        Create a zip file containing the images specified by the given image names or board id.
        If download with the same bulk_download_id already exists, it will be overwritten.

        :return: The name of the zip file.
        """
        zip_file_name = bulk_download_item_id + ".zip"
        zip_file_path = self._bulk_downloads_folder / (zip_file_name)

        with ZipFile(zip_file_path, "w") as zip_file:
            for image_dto in image_dtos:
                image_zip_path = Path(image_dto.image_category.value) / image_dto.image_name
                image_disk_path = self._invoker.services.images.get_path(image_dto.image_name)
                zip_file.write(image_disk_path, arcname=image_zip_path)

        return str(zip_file_name)

    # from https://stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string
    def _clean_string_to_path_safe(self, s: str) -> str:
        """Clean a string to be path safe."""
        return "".join([c for c in s if c.isalpha() or c.isdigit() or c == " " or c == "_" or c == "-"]).rstrip()

    def _signal_job_started(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        """Signal that a bulk download job has started."""
        if self._event_bus:
            assert bulk_download_id is not None
            self._event_bus.emit_bulk_download_started(
                bulk_download_id=bulk_download_id,
                bulk_download_item_id=bulk_download_item_id,
                bulk_download_item_name=bulk_download_item_name,
            )

    def _signal_job_completed(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        """Signal that a bulk download job has completed."""
        if self._event_bus:
            assert bulk_download_id is not None
            assert bulk_download_item_name is not None
            self._event_bus.emit_bulk_download_completed(
                bulk_download_id=bulk_download_id,
                bulk_download_item_id=bulk_download_item_id,
                bulk_download_item_name=bulk_download_item_name,
            )

    def _signal_job_failed(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str, exception: Exception
    ) -> None:
        """Signal that a bulk download job has failed."""
        if self._event_bus:
            assert bulk_download_id is not None
            assert exception is not None
            self._event_bus.emit_bulk_download_failed(
                bulk_download_id=bulk_download_id,
                bulk_download_item_id=bulk_download_item_id,
                bulk_download_item_name=bulk_download_item_name,
                error=str(exception),
            )

    def stop(self, *args, **kwargs):
        self._temp_directory.cleanup()

    def delete(self, bulk_download_item_name: str) -> None:
        path = self.get_path(bulk_download_item_name)
        Path(path).unlink()

    def get_path(self, bulk_download_item_name: str) -> str:
        path = str(self._bulk_downloads_folder / bulk_download_item_name)
        if not self._is_valid_path(path):
            raise BulkDownloadTargetException()
        return path

    def _is_valid_path(self, path: Union[str, Path]) -> bool:
        """Validates the path given for a bulk download."""
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()
