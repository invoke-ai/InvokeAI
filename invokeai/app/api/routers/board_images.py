from fastapi import Body, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies

board_images_router = APIRouter(prefix="/v1/board_images", tags=["boards"])


class AddImagesToBoardResult(BaseModel):
    board_id: str = Field(description="The id of the board the images were added to")
    added_image_names: list[str] = Field(description="The image names that were added to the board")


class RemoveImagesFromBoardResult(BaseModel):
    removed_image_names: list[str] = Field(description="The image names that were removed from their board")


@board_images_router.post(
    "/",
    operation_id="add_image_to_board",
    responses={
        201: {"description": "The image was added to a board successfully"},
    },
    status_code=201,
)
async def add_image_to_board(
    board_id: str = Body(description="The id of the board to add to"),
    image_name: str = Body(description="The name of the image to add"),
):
    """Creates a board_image"""
    try:
        result = ApiDependencies.invoker.services.board_images.add_image_to_board(
            board_id=board_id, image_name=image_name
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add image to board")


@board_images_router.delete(
    "/",
    operation_id="remove_image_from_board",
    responses={
        201: {"description": "The image was removed from the board successfully"},
    },
    status_code=201,
)
async def remove_image_from_board(
    image_name: str = Body(description="The name of the image to remove", embed=True),
):
    """Removes an image from its board, if it had one"""
    try:
        result = ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove image from board")


@board_images_router.post(
    "/batch",
    operation_id="add_images_to_board",
    responses={
        201: {"description": "Images were added to board successfully"},
    },
    status_code=201,
    response_model=AddImagesToBoardResult,
)
async def add_images_to_board(
    board_id: str = Body(description="The id of the board to add to"),
    image_names: list[str] = Body(description="The names of the images to add", embed=True),
) -> AddImagesToBoardResult:
    """Adds a list of images to a board"""
    try:
        added_image_names: list[str] = []
        for image_name in image_names:
            try:
                ApiDependencies.invoker.services.board_images.add_image_to_board(
                    board_id=board_id, image_name=image_name
                )
                added_image_names.append(image_name)
            except Exception:
                pass
        return AddImagesToBoardResult(board_id=board_id, added_image_names=added_image_names)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add images to board")


@board_images_router.post(
    "/batch/delete",
    operation_id="remove_images_from_board",
    responses={
        201: {"description": "Images were removed from board successfully"},
    },
    status_code=201,
    response_model=RemoveImagesFromBoardResult,
)
async def remove_images_from_board(
    image_names: list[str] = Body(description="The names of the images to remove", embed=True),
) -> RemoveImagesFromBoardResult:
    """Removes a list of images from their board, if they had one"""
    try:
        removed_image_names: list[str] = []
        for image_name in image_names:
            try:
                ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
                removed_image_names.append(image_name)
            except Exception:
                pass
        return RemoveImagesFromBoardResult(removed_image_names=removed_image_names)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove images from board")
