from pathlib import Path
from typing import Optional, Union

from abc import ABC, abstractmethod

from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker

class BulkDownloadBase(ABC):

    @abstractmethod
    def __init__(
        self,
        output_folder: Union[str, Path],
        event_bus: Optional["EventServiceBase"] = None,
    ):
        """
        Create BulkDownloadBase object.

        :param output_folder: The path to the output folder where the bulk download files can be temporarily stored.
        :param event_bus: InvokeAI event bus for reporting events to.
        """

    @abstractmethod
    def handler(self, invoker: Invoker, image_names: list[str], board_id: Optional[str]) -> None:
        """
        Starts a a bulk download job.
        
        :param invoker: The Invoker that holds all the services, required to be passed as a parameter to avoid circular dependencies.
        :param image_names: A list of image names to include in the zip file.
        :param board_id: The ID of the board. If provided, all images associated with the board will be included in the zip file.
        """