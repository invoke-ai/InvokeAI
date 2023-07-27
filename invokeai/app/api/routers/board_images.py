from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.services.board_record_storage import BoardRecord, BoardChanges
from invokeai.app.services.image_record_storage import OffsetPaginatedResults
from invokeai.app.services.models.board_record import BoardDTO
from invokeai.app.services.models.image_record import ImageDTO

from ..dependencies import ApiDependencies

board_images_router = APIRouter(prefix="/v1/board_images", tags=["boards"])


@board_images_router.post(
    "/",
    operation_id="create_board_image",
    responses={
        201: {"description": "The image was added to a board successfully"},
    },
    status_code=201,
)
async def create_board_image(
    board_id: str = Body(description="The id of the board to add to"),
    image_name: str = Body(description="The name of the image to add"),
):
    """Creates a board_image"""
    try:
        result = ApiDependencies.invoker.services.board_images.add_image_to_board(
            board_id=board_id, image_name=image_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to add to board")


@board_images_router.delete(
    "/",
    operation_id="remove_board_image",
    responses={
        201: {"description": "The image was removed from the board successfully"},
    },
    status_code=201,
)
async def remove_board_image(
    board_id: str = Body(description="The id of the board"),
    image_name: str = Body(description="The name of the image to remove"),
):
    """Deletes a board_image"""
    try:
        result = ApiDependencies.invoker.services.board_images.remove_image_from_board(
            board_id=board_id, image_name=image_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update board")
