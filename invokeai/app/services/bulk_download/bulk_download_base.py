from abc import ABC, abstractmethod
from typing import Optional


class BulkDownloadBase(ABC):
    """Responsible for creating a zip file containing the images specified by the given image names or board id."""

    @abstractmethod
    def handler(
        self, image_names: Optional[list[str]], board_id: Optional[str], bulk_download_item_id: Optional[str]
    ) -> None:
        """
        Create a zip file containing the images specified by the given image names or board id.

        :param image_names: A list of image names to include in the zip file.
        :param board_id: The ID of the board. If provided, all images associated with the board will be included in the zip file.
        :param bulk_download_item_id: The bulk_download_item_id that will be used to retrieve the bulk download item when it is prepared, if none is provided a uuid will be generated.
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
