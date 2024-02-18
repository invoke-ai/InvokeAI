from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.invoker import Invoker


class BulkDownloadBase(ABC):
    @abstractmethod
    def start(self, invoker: Invoker) -> None:
        """
        Starts the BulkDownloadService.

        This method is responsible for starting the BulkDownloadService and performing any necessary initialization
        operations to prepare the service for use.

        param: invoker: The Invoker that holds all the services, required to be passed as a parameter to avoid circular dependencies.
        """

    @abstractmethod
    def __init__(self):
        """
        Create BulkDownloadBase object.
        """

    @abstractmethod
    def handler(
        self, image_names: Optional[list[str]], board_id: Optional[str], bulk_download_item_id: Optional[str]
    ) -> None:
        """
        Starts a a bulk download job.

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
    def stop(self, *args, **kwargs) -> None:
        """
        Stops the BulkDownloadService and cleans up all the remnants.

        This method is responsible for stopping the BulkDownloadService and performing any necessary cleanup
        operations to remove any remnants or resources associated with the service.

        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """

    @abstractmethod
    def delete(self, bulk_download_item_name: str) -> None:
        """
        Delete the bulk download file.

        :param bulk_download_item_name: The name of the bulk download item.
        """
