import uuid
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.bulk_download.bulk_download_common import BulkDownloadException
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException
from invokeai.app.services.invoker import Invoker

from .bulk_download_base import BulkDownloadBase


class BulkDownloadService(BulkDownloadBase):
    __output_folder: Path
    __bulk_downloads_folder: Path
    __event_bus: Optional[EventServiceBase]

    def __init__(
        self,
        output_folder: Union[str, Path],
        event_bus: Optional[EventServiceBase] = None,
    ):
        """
        Initialize the downloader object.

        :param event_bus: Optional EventService object
        """
        self.__output_folder: Path = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__bulk_downloads_folder = self.__output_folder / "bulk_downloads"
        self.__bulk_downloads_folder.mkdir(parents=True, exist_ok=True)
        self.__event_bus = event_bus

    def handler(self, invoker: Invoker, image_names: list[str], board_id: Optional[str]) -> None:
        """
        Create a zip file containing the images specified by the given image names or board id.

        param: image_names: A list of image names to include in the zip file.
        param: board_id: The ID of the board. If provided, all images associated with the board will be included in the zip file.
        """
        bulk_download_id = str(uuid.uuid4())

        try:
            if board_id:
                image_names = invoker.services.board_image_records.get_all_board_image_names_for_board(board_id)
                if board_id == "none":
                    board_id = "Uncategorized"
            image_names_to_paths: dict[str, str] = self._get_image_name_to_path_map(invoker, image_names)
            file_path: str = self._create_zip_file(image_names_to_paths, bulk_download_id)
            self._signal_job_completed(bulk_download_id, file_path)
        except (ImageRecordNotFoundException, BoardRecordNotFoundException, BulkDownloadException) as e:
            self._signal_job_failed(bulk_download_id, e)
        except Exception as e:
            self._signal_job_failed(bulk_download_id, e)

    def _get_image_name_to_path_map(self, invoker: Invoker, image_names: list[str]) -> dict[str, str]:
        """
        Create a map of image names to their paths.
        :param image_names: A list of image names.
        """
        image_names_to_paths: dict[str, str] = {}
        for image_name in image_names:
            image_names_to_paths[image_name] = invoker.services.images.get_path(image_name)
        return image_names_to_paths

    def _create_zip_file(self, image_names_to_paths: dict[str, str], bulk_download_id: str) -> str:
        """
        Create a zip file containing the images specified by the given image names or board id.
        If download with the same bulk_download_id already exists, it will be overwritten.
        """

        zip_file_path = self.__bulk_downloads_folder / (bulk_download_id + ".zip")

        with ZipFile(zip_file_path, "w") as zip_file:
            for image_name, image_path in image_names_to_paths.items():
                zip_file.write(image_path, arcname=image_name)

        return str(zip_file_path)

    def _signal_job_started(self, bulk_download_id: str) -> None:
        """Signal that a bulk download job has started."""
        if self.__event_bus:
            assert bulk_download_id is not None
            self.__event_bus.emit_bulk_download_started(
                bulk_download_id=bulk_download_id,
            )

    def _signal_job_completed(self, bulk_download_id: str, file_path: str) -> None:
        """Signal that a bulk download job has completed."""
        if self.__event_bus:
            assert bulk_download_id is not None
            assert file_path is not None
            self.__event_bus.emit_bulk_download_completed(
                bulk_download_id=bulk_download_id,
                file_path=file_path,
            )

    def _signal_job_failed(self, bulk_download_id: str, exception: Exception) -> None:
        """Signal that a bulk download job has failed."""
        if self.__event_bus:
            assert bulk_download_id is not None
            assert exception is not None
            self.__event_bus.emit_bulk_download_failed(
                bulk_download_id=bulk_download_id,
                error=str(exception),
            )
