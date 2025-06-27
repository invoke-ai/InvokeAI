from fastapi import Body, HTTPException
from fastapi.routing import APIRouter

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.images.images_common import AddImagesToBoardResult, RemoveImagesFromBoardResult

board_images_router = APIRouter(prefix="/v1/board_images", tags=["boards"])


@board_images_router.post(
    "/",
    operation_id="add_image_to_board",
    responses={
        201: {"description": "The image was added to a board successfully"},
    },
    status_code=201,
    response_model=AddImagesToBoardResult,
)
async def add_image_to_board(
    board_id: str = Body(description="The id of the board to add to"),
    image_name: str = Body(description="The name of the image to add"),
) -> AddImagesToBoardResult:
    """Creates a board_image"""
    try:
        added_images: set[str] = set()
        affected_boards: set[str] = set()
        old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
        ApiDependencies.invoker.services.board_images.add_image_to_board(board_id=board_id, image_name=image_name)
        added_images.add(image_name)
        affected_boards.add(board_id)
        affected_boards.add(old_board_id)

        return AddImagesToBoardResult(
            added_images=list(added_images),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add image to board")


@board_images_router.delete(
    "/",
    operation_id="remove_image_from_board",
    responses={
        201: {"description": "The image was removed from the board successfully"},
    },
    status_code=201,
    response_model=RemoveImagesFromBoardResult,
)
async def remove_image_from_board(
    image_name: str = Body(description="The name of the image to remove", embed=True),
) -> RemoveImagesFromBoardResult:
    """Removes an image from its board, if it had one"""
    try:
        removed_images: set[str] = set()
        affected_boards: set[str] = set()
        old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
        ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
        removed_images.add(image_name)
        affected_boards.add("none")
        affected_boards.add(old_board_id)
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            affected_boards=list(affected_boards),
        )

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
        added_images: set[str] = set()
        affected_boards: set[str] = set()
        for image_name in image_names:
            try:
                old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
                ApiDependencies.invoker.services.board_images.add_image_to_board(
                    board_id=board_id,
                    image_name=image_name,
                )
                added_images.add(image_name)
                affected_boards.add(board_id)
                affected_boards.add(old_board_id)

            except Exception:
                pass
        return AddImagesToBoardResult(
            added_images=list(added_images),
            affected_boards=list(affected_boards),
        )
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
        removed_images: set[str] = set()
        affected_boards: set[str] = set()
        for image_name in image_names:
            try:
                old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
                ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
                removed_images.add(image_name)
                affected_boards.add("none")
                affected_boards.add(old_board_id)
            except Exception:
                pass
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove images from board")
