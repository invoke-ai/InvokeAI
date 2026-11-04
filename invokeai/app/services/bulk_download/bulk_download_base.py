from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.bulk_download.bulk_download_common import BulkDownloadStatus
from invokeai.app.services.images.images_common import ImageExportFormat


class BulkDownloadBase(ABC):
    """Responsible for creating a zip file containing the images specified by the given image names or board id."""

    @abstractmethod
    def start_job(
        self,
        image_names: Optional[list[str]],
        board_id: Optional[str],
        bulk_download_item_id: str,
        user_id: str,
        export_format: Optional[ImageExportFormat] = None,
    ) -> None:
        """
        Register the download, so the fetch endpoint can answer for it at once (building, owner), and build it on a
        thread the service owns, which `stop` ends between two images. Raises BulkDownloadBusyError for a `png16`
        download while another one is being built.
        """

    @abstractmethod
    def get_status(self, bulk_download_item_name: str) -> BulkDownloadStatus:
        """Whether the archive is ready, still being built, failed (see `pop_failure`) or unknown."""

    @abstractmethod
    def pop_failure(self, bulk_download_item_name: str) -> Optional[str]:
        """The reason a download failed, once; the download is forgotten afterwards."""

    @abstractmethod
    def handler(
        self,
        image_names: Optional[list[str]],
        board_id: Optional[str],
        bulk_download_item_id: Optional[str],
        user_id: str = "system",
        export_format: Optional[ImageExportFormat] = None,
    ) -> None:
        """
        Create a zip file containing the images specified by the given image names or board id.

        :param image_names: A list of image names to include in the zip file.
        :param board_id: The ID of the board. If provided, all images associated with the board will be included in the zip file.
        :param bulk_download_item_id: The bulk_download_item_id that will be used to retrieve the bulk download item when it is prepared, if none is provided a uuid will be generated.
        :param user_id: The ID of the user who initiated the download.
        :param export_format: With `png16`, every image with an exportable precise asset is stored as
            `<category>/<stem>-16bit.png` (the bytes of the single-image export) instead of its gallery file.
            An image whose export fails keeps its gallery file and one warning is logged. None stores the
            gallery files.
        """

    @abstractmethod
    def get_path(self, bulk_download_item_name: str) -> str:
        """
        Get the path to the bulk download file.

        :param bulk_download_item_name: The name of the bulk download item.
        :return: The path to the bulk download file.
        """

    @abstractmethod
    def generate_item_id(self, board_id: Optional[str]) -> str:
        """
        Generate an item ID for a bulk download item.

        :param board_id: The ID of the board whose name is to be included in the item id.
        :return: The generated item ID.
        """

    @abstractmethod
    def delete(self, bulk_download_item_name: str) -> None:
        """
        Delete the bulk download file.

        :param bulk_download_item_name: The name of the bulk download item.
        """

    @abstractmethod
    def get_owner(self, bulk_download_item_name: str) -> Optional[str]:
        """
        Get the user_id of the user who initiated the download.

        :param bulk_download_item_name: The name of the bulk download item.
        :return: The user_id of the owner, or None if not tracked.
        """
