import uuid
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.bulk_download.bulk_download_common import (
    DEFAULT_BULK_DOWNLOAD_ID,
    BulkDownloadException,
    BulkDownloadTargetException,
)
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordNotFoundException
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

    def get_path(self, bulk_download_item_name: str) -> str:
        """
        Get the path to the bulk download file.

        :param bulk_download_item_name: The name of the bulk download item.
        :return: The path to the bulk download file.
        """
        path = str(self.__bulk_downloads_folder / bulk_download_item_name)
        if not self.validate_path(path):
            raise BulkDownloadTargetException()
        return path

    def validate_path(self, path: Union[str, Path]) -> bool:
        """Validates the path given for a bulk download."""
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    def handler(self, invoker: Invoker, image_names: list[str], board_id: Optional[str]) -> None:
        """
        Create a zip file containing the images specified by the given image names or board id.

        param: image_names: A list of image names to include in the zip file.
        param: board_id: The ID of the board. If provided, all images associated with the board will be included in the zip file.
        """

        bulk_download_id: str = DEFAULT_BULK_DOWNLOAD_ID
        bulk_download_item_id: str = str(uuid.uuid4()) if board_id is None else board_id

        try:
            board_name: str = ""
            if board_id:
                # -1 is the default value for limit, which means no limit, is_intermediate only gives us completed images
                image_names = [
                    img.image_name
                    for img in invoker.services.images.get_many(
                        offset=0,
                        limit=-1,
                        board_id=board_id,
                        is_intermediate=False,
                        categories=[ImageCategory.GENERAL],
                    ).items
                ]
                if board_id == "none":
                    board_id = "Uncategorized"
                    board_name = "Uncategorized"
                else:
                    board_name = invoker.services.board_records.get(board_id).board_name
                board_name = self._clean_string_to_path_safe(board_name)

            self._signal_job_started(bulk_download_id, bulk_download_item_id)

            image_names_to_paths: dict[str, str] = self._get_image_name_to_path_map(invoker, image_names)
            bulk_download_item_name: str = self._create_zip_file(
                image_names_to_paths, bulk_download_item_id if board_id is None else board_name
            )
            self._signal_job_completed(bulk_download_id, bulk_download_item_id, bulk_download_item_name)
        except (ImageRecordNotFoundException, BoardRecordNotFoundException, BulkDownloadException) as e:
            self._signal_job_failed(bulk_download_id, bulk_download_item_id, e)
        except Exception as e:
            self._signal_job_failed(bulk_download_id, bulk_download_item_id, e)

    def _get_image_name_to_path_map(self, invoker: Invoker, image_names: list[str]) -> dict[str, str]:
        """
        Create a map of image names to their paths.
        :param image_names: A list of image names.
        """
        image_names_to_paths: dict[str, str] = {}
        for image_name in image_names:
            image_names_to_paths[image_name] = invoker.services.images.get_path(image_name)
        return image_names_to_paths

    def _create_zip_file(self, image_names_to_paths: dict[str, str], bulk_download_item_id: str) -> str:
        """
        Create a zip file containing the images specified by the given image names or board id.
        If download with the same bulk_download_id already exists, it will be overwritten.

        :return: The name of the zip file.
        """
        zip_file_name = bulk_download_item_id + ".zip"
        zip_file_path = self.__bulk_downloads_folder / (zip_file_name)

        with ZipFile(zip_file_path, "w") as zip_file:
            for image_name, image_path in image_names_to_paths.items():
                zip_file.write(image_path, arcname=image_name)

        return str(zip_file_name)

    def _clean_string_to_path_safe(self, s: str) -> str:
        """Clean a string to be path safe."""
        return "".join([c for c in s if c.isalpha() or c.isdigit() or c == " "]).rstrip()

    def _signal_job_started(self, bulk_download_id: str, bulk_download_item_id: str) -> None:
        """Signal that a bulk download job has started."""
        if self.__event_bus:
            assert bulk_download_id is not None
            self.__event_bus.emit_bulk_download_started(
                bulk_download_id=bulk_download_id,
                bulk_download_item_id=bulk_download_item_id,
            )

    def _signal_job_completed(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        """Signal that a bulk download job has completed."""
        if self.__event_bus:
            assert bulk_download_id is not None
            assert bulk_download_item_name is not None
            self.__event_bus.emit_bulk_download_completed(
                bulk_download_id=bulk_download_id,
                bulk_download_item_id=bulk_download_item_id,
                bulk_download_item_name=bulk_download_item_name,
            )

    def _signal_job_failed(self, bulk_download_id: str, bulk_download_item_id: str, exception: Exception) -> None:
        """Signal that a bulk download job has failed."""
        if self.__event_bus:
            assert bulk_download_id is not None
            assert exception is not None
            self.__event_bus.emit_bulk_download_failed(
                bulk_download_id=bulk_download_id,
                bulk_download_item_id=bulk_download_item_id,
                error=str(exception),
            )

    def stop(self, *args, **kwargs):
        """Stop the bulk download service and delete the files in the bulk download folder."""
        # Get all the files in the bulk downloads folder
        files = self.__bulk_downloads_folder.glob("*")

        # Delete all the files
        for file in files:
            file.unlink()
