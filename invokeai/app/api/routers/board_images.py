from fastapi import Body, HTTPException
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.images.images_common import AddImagesToBoardResult, RemoveImagesFromBoardResult

board_images_router = APIRouter(prefix="/v1/board_images", tags=["boards"])


def _assert_board_write_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not mutate the given board.

    Write access is granted when ANY of these hold:
    - The user is an admin.
    - The user owns the board.
    - The board visibility is Public (public boards accept contributions from any user).
    """
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")
    if current_user.is_admin:
        return
    if board.user_id == current_user.user_id:
        return
    if board.board_visibility == BoardVisibility.Public:
        return
    raise HTTPException(status_code=403, detail="Not authorized to modify this board")


def _assert_image_direct_owner(image_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user is not the direct owner of the image.

    This is intentionally stricter than _assert_image_owner in images.py:
    board ownership is NOT sufficient here.  Allowing a user to add someone
    else's image to their own board would grant them mutation rights via the
    board-ownership fallback in _assert_image_owner, escalating read access
    into write access.
    """
    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.image_records.get_user_id(image_name)
    if owner is not None and owner == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to move this image")


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
    current_user: CurrentUserOrDefault,
    board_id: str = Body(description="The id of the board to add to"),
    image_name: str = Body(description="The name of the image to add"),
) -> AddImagesToBoardResult:
    """Creates a board_image"""
    _assert_board_write_access(board_id, current_user)
    _assert_image_direct_owner(image_name, current_user)
    try:
        added_images: set[str] = set()
        affected_boards: set[str] = set()
        old_board_id = ApiDependencies.invoker.services.board_image_records.get_board_for_image(image_name) or "none"
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
    current_user: CurrentUserOrDefault,
    image_name: str = Body(description="The name of the image to remove", embed=True),
) -> RemoveImagesFromBoardResult:
    """Removes an image from its board, if it had one"""
    try:
        old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
        if old_board_id != "none":
            _assert_board_write_access(old_board_id, current_user)
        removed_images: set[str] = set()
        affected_boards: set[str] = set()
        ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
        removed_images.add(image_name)
        affected_boards.add("none")
        affected_boards.add(old_board_id)
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            affected_boards=list(affected_boards),
        )

    except HTTPException:
        raise
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
    current_user: CurrentUserOrDefault,
    board_id: str = Body(description="The id of the board to add to"),
    image_names: list[str] = Body(description="The names of the images to add", embed=True),
) -> AddImagesToBoardResult:
    """Adds a list of images to a board"""
    _assert_board_write_access(board_id, current_user)
    try:
        added_images: set[str] = set()
        affected_boards: set[str] = set()
        for image_name in image_names:
            try:
                _assert_image_direct_owner(image_name, current_user)
                old_board_id = (
                    ApiDependencies.invoker.services.board_image_records.get_board_for_image(image_name) or "none"
                )
                ApiDependencies.invoker.services.board_images.add_image_to_board(
                    board_id=board_id,
                    image_name=image_name,
                )
                added_images.add(image_name)
                affected_boards.add(board_id)
                affected_boards.add(old_board_id)

            except HTTPException:
                raise
            except Exception:
                pass
        return AddImagesToBoardResult(
            added_images=list(added_images),
            affected_boards=list(affected_boards),
        )
    except HTTPException:
        raise
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
    current_user: CurrentUserOrDefault,
    image_names: list[str] = Body(description="The names of the images to remove", embed=True),
) -> RemoveImagesFromBoardResult:
    """Removes a list of images from their board, if they had one"""
    try:
        removed_images: set[str] = set()
        affected_boards: set[str] = set()
        for image_name in image_names:
            try:
                old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
                if old_board_id != "none":
                    _assert_board_write_access(old_board_id, current_user)
                ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
                removed_images.add(image_name)
                affected_boards.add("none")
                affected_boards.add(old_board_id)
            except HTTPException:
                raise
            except Exception:
                pass
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            affected_boards=list(affected_boards),
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove images from board")
