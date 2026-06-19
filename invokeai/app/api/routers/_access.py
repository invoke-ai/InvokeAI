"""Cross-router authorization helpers.

These helpers are imported by multiple router modules. Keep them free of router
specifics so any route can call them after resolving `current_user`.
"""

from fastapi import HTTPException

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_records.board_records_common import BoardVisibility


def assert_image_owner(image_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user does not own the image and is not an admin.

    Ownership is satisfied when ANY of these hold:
    - The user is an admin.
    - The user is the image's direct owner (image_records.user_id).
    - The user owns the board the image sits on.
    - The image sits on a Public board (public boards grant mutation rights).
    """
    if current_user.is_admin:
        return
    if not ApiDependencies.invoker.services.image_records.exists(image_name):
        raise HTTPException(status_code=404, detail="Image not found")
    owner = ApiDependencies.invoker.services.image_records.get_user_id(image_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_image_records.get_board_for_image(image_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.user_id == current_user.user_id:
                return
            if board.board_visibility == BoardVisibility.Public:
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to modify this image")


def assert_image_read_access(image_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not view the image.

    Access is granted when ANY of these hold:
    - The user is an admin.
    - The user owns the image.
    - The image sits on a shared or public board.
    """
    if current_user.is_admin:
        return
    if not ApiDependencies.invoker.services.image_records.exists(image_name):
        raise HTTPException(status_code=404, detail="Image not found")

    owner = ApiDependencies.invoker.services.image_records.get_user_id(image_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_image_records.get_board_for_image(image_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to access this image")


def assert_board_read_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not read images from this board.

    Access is granted when ANY of these hold:
    - The user is an admin.
    - The user owns the board.
    - The board visibility is Shared or Public.
    """
    if current_user.is_admin:
        return

    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if board.user_id == current_user.user_id:
        return

    if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
        return

    raise HTTPException(status_code=403, detail="Not authorized to access this board")
