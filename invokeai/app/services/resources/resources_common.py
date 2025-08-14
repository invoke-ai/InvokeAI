from enum import Enum

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    """The type of resource that can be associated with a board."""

    IMAGE = "image"
    VIDEO = "video"


class ResourceIdentifier(BaseModel):
    resource_id: str = Field(description="The id of the resource to delete")
    resource_type: ResourceType = Field(description="The type of the resource to delete")


class ResultWithAffectedBoards(BaseModel):
    affected_boards: list[str] = Field(description="The ids of boards affected by the delete operation")


class DeleteResourcesResult(ResultWithAffectedBoards):
    deleted_resources: list[ResourceIdentifier] = Field(description="The ids of the resources that were deleted")


class StarredResourcesResult(ResultWithAffectedBoards):
    starred_resources: list[ResourceIdentifier] = Field(description="The resources that were starred")


class UnstarredResourcesResult(ResultWithAffectedBoards):
    unstarred_resources: list[ResourceIdentifier] = Field(description="The resources that were unstarred")


class AddResourcesToBoardResult(ResultWithAffectedBoards):
    added_resources: list[ResourceIdentifier] = Field(description="The resources that were added to the board")


class RemoveResourcesFromBoardResult(ResultWithAffectedBoards):
    removed_resources: list[ResourceIdentifier] = Field(description="The resources that were removed from their board")
