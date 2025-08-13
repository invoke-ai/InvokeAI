from fastapi import Body, HTTPException
from fastapi.routing import APIRouter

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_resources.board_resources_common import ResourceType
from invokeai.app.services.resources.resources_common import AddResourcesToBoardResult, RemoveResourcesFromBoardResult

board_resources_router = APIRouter(prefix="/v1/board_resources", tags=["boards"])


@board_resources_router.post(
    "/",
    operation_id="add_resource_to_board",
    responses={
        201: {"description": "The resource was added to a board successfully"},
    },
    status_code=201,
    response_model=AddResourcesToBoardResult, 
)
async def add_resource_to_board(
    board_id: str = Body(description="The id of the board to add to"),
    resource_id: str = Body(description="The id of the resource to add"),
    resource_type: ResourceType = Body(description="The type of resource"),
) -> AddResourcesToBoardResult:
    """Creates a board_resource relationship"""
    try:
        added_resources: set[str] = set()
        affected_boards: set[str] = set()
        
        if resource_type == ResourceType.IMAGE:
            old_board_id = ApiDependencies.invoker.services.images.get_dto(resource_id).board_id or "none"
        else:
            # For videos, we'll need to implement this once video service exists
            old_board_id = "none"
            
        ApiDependencies.invoker.services.board_resources.add_resource_to_board(
            board_id=board_id, 
            resource_id=resource_id, 
            resource_type=resource_type
        )
        added_resources.add(resource_id)
        affected_boards.add(board_id)
        affected_boards.add(old_board_id)

        return AddResourcesToBoardResult(
            added_resources=list(added_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add resource to board")


@board_resources_router.delete(
    "/",
    operation_id="remove_resource_from_board",
    responses={
        201: {"description": "The resource was removed from the board successfully"},
    },
    status_code=201,
    response_model=RemoveResourcesFromBoardResult,  # For now, using same response model
)
async def remove_resource_from_board(
    resource_id: str = Body(description="The id of the resource to remove", embed=True),
    resource_type: ResourceType = Body(description="The type of resource", embed=True),
) -> RemoveResourcesFromBoardResult:
    """Removes a resource from its board, if it had one"""
    try:
        removed_resources: set[str] = set()
        affected_boards: set[str] = set()
        
        if resource_type == ResourceType.IMAGE:
            old_board_id = ApiDependencies.invoker.services.images.get_dto(resource_id).board_id or "none"
        else:
            # For videos, we'll need to implement this once video service exists
            old_board_id = "none"
            
        ApiDependencies.invoker.services.board_resources.remove_resource_from_board(
            resource_id=resource_id, 
            resource_type=resource_type
        )
        removed_resources.add(resource_id)
        affected_boards.add("none")
        affected_boards.add(old_board_id)
        
        return RemoveResourcesFromBoardResult(
            removed_resources=list(removed_resources),
            affected_boards=list(affected_boards),
        )

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove resource from board")


@board_resources_router.post(
    "/batch",
    operation_id="add_resources_to_board",
    responses={
        201: {"description": "Resources were added to board successfully"},
    },
    status_code=201,
    response_model=AddResourcesToBoardResult,  # For now, using same response model
)
async def add_resources_to_board(
    board_id: str = Body(description="The id of the board to add to"),
    resource_ids: list[str] = Body(description="The ids of the resources to add", embed=True),
    resource_type: ResourceType = Body(description="The type of resources"),
) -> AddResourcesToBoardResult:
    """Adds a list of resources to a board"""
    try:
        added_resources: set[str] = set()
        affected_boards: set[str] = set()
        for resource_id in resource_ids:
            try:
                if resource_type == ResourceType.IMAGE:
                    old_board_id = ApiDependencies.invoker.services.images.get_dto(resource_id).board_id or "none"
                else:
                    # For videos, we'll need to implement this once video service exists
                    old_board_id = "none"
                    
                ApiDependencies.invoker.services.board_resources.add_resource_to_board(
                    board_id=board_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                )
                added_resources.add(resource_id)
                affected_boards.add(board_id)
                affected_boards.add(old_board_id)

            except Exception:
                pass
        return AddResourcesToBoardResult(
            added_resources=list(added_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add resources to board")


@board_resources_router.post(
    "/batch/delete",
    operation_id="remove_resources_from_board",
    responses={
        201: {"description": "Resources were removed from board successfully"},
    },
    status_code=201,
    response_model=RemoveResourcesFromBoardResult,  # For now, using same response model
)
async def remove_resources_from_board(
    resource_ids: list[str] = Body(description="The ids of the resources to remove", embed=True),
    resource_type: ResourceType = Body(description="The type of resources", embed=True),
) -> RemoveResourcesFromBoardResult:
    """Removes a list of resources from their board, if they had one"""
    try:
        removed_resources: set[str] = set()
        affected_boards: set[str] = set()
        for resource_id in resource_ids:
            try:
                if resource_type == ResourceType.IMAGE:
                    old_board_id = ApiDependencies.invoker.services.images.get_dto(resource_id).board_id or "none"
                else:
                    # For videos, we'll need to implement this once video service exists
                    old_board_id = "none"
                    
                ApiDependencies.invoker.services.board_resources.remove_resource_from_board(
                    resource_id=resource_id, 
                    resource_type=resource_type
                )
                removed_resources.add(resource_id)
                affected_boards.add("none")
                affected_boards.add(old_board_id)
            except Exception:
                pass
        return RemoveResourcesFromBoardResult(
            removed_resources=list(removed_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove resources from board")

