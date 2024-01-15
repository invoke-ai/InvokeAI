from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

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
    def __init__(self, output_folder: Union[str, Path]):
        """
        Create BulkDownloadBase object.

        :param output_folder: The path to the output folder where the bulk download files can be temporarily stored.
        """

    @abstractmethod
    def handler(self, image_names: list[str], board_id: Optional[str]) -> None:
        """
        Starts a a bulk download job.

        :param invoker: The Invoker that holds all the services, required to be passed as a parameter to avoid circular dependencies.
        :param image_names: A list of image names to include in the zip file.
        :param board_id: The ID of the board. If provided, all images associated with the board will be included in the zip file.
        """

    @abstractmethod
    def get_path(self, bulk_download_item_name: str) -> str:
        """
        Get the path to the bulk download file.

        :param bulk_download_item_name: The name of the bulk download item.
        :return: The path to the bulk download file.
        """

    @abstractmethod
    def stop(self, *args, **kwargs) -> None:
        """
        Stops the BulkDownloadService and cleans up all the remnants.

        This method is responsible for stopping the BulkDownloadService and performing any necessary cleanup
        operations to remove any remnants or resources associated with the service.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
