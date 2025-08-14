from pydantic import Field

from invokeai.app.services.resources.resources_common import ResourceType
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class BoardResource(BaseModelExcludeNull):
    """Represents a resource (image or video) associated with a board."""

    board_id: str = Field(description="The id of the board")
    resource_id: str = Field(description="The id of the resource (image_name or video_id)")
    resource_type: ResourceType = Field(description="The type of resource")
