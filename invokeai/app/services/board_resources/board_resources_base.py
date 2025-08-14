from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.resources.resources_common import ResourceIdentifier, ResourceType


class BoardResourcesServiceABC(ABC):
    """High-level service for board-resource relationship management."""

    @abstractmethod
    def add_resource_to_board(
        self,
        board_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Adds a resource (image or video) to a board."""
        pass

    @abstractmethod
    def remove_resource_from_board(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Removes a resource (image or video) from a board."""
        pass

    @abstractmethod
    def get_all_board_resource_ids_for_board(
        self,
        board_id: str,
        resource_type: Optional[ResourceType] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> list[ResourceIdentifier]:
        """Gets all board resources for a board, as a list of resource IDs."""
        pass

    @abstractmethod
    def get_board_for_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> Optional[str]:
        """Gets a resource's board id, if it has one."""
        pass
