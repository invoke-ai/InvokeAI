from typing import Optional

from invokeai.app.services.board_resources.board_resources_base import BoardResourcesServiceABC
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.resources.resources_common import ResourceIdentifier, ResourceType


class BoardResourcesService(BoardResourcesServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def add_resource_to_board(
        self,
        board_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        self.__invoker.services.board_resource_records.add_resource_to_board(board_id, resource_id, resource_type)

    def remove_resource_from_board(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        self.__invoker.services.board_resource_records.remove_resource_from_board(resource_id, resource_type)

    def get_all_board_resource_ids_for_board(
        self,
        board_id: str,
        resource_type: Optional[ResourceType] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
    ) -> list[ResourceIdentifier]:
        return self.__invoker.services.board_resource_records.get_all_board_resource_ids_for_board(
            board_id,
            resource_type,
            categories,
            is_intermediate,
        )

    def get_board_for_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> Optional[str]:
        return self.__invoker.services.board_resource_records.get_board_for_resource(resource_id, resource_type)
