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
    - The user owns the board the image sits on.
    - The image sits on a shared or public board.
    - The image sits on a board explicitly shared with the user.

    Board-backed images defer to `assert_board_read_access` so individual image
    reads stay consistent with board listings (including board_id="all").
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
        assert_board_read_access(board_id, current_user)
        return

    raise HTTPException(status_code=403, detail="Not authorized to access this image")


def assert_board_write_access(board_id: str | None, current_user: CurrentUserOrDefault) -> None:
    """Raise if the current user may not put media on this board.

    `None` means "no board" — always allowed, so upload routes can pass their optional board
    straight through. Otherwise access is granted when the user is an admin, owns the board, or
    the board is Public (public boards accept contributions from any user).

    Shared boards are deliberately read-only here: `Shared` grants visibility, not contribution.
    """
    if board_id is None:
        return

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


def assert_video_read_access(video_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not view the video.

    Deliberately identical in shape to `assert_image_read_access`, and delegating to
    `assert_board_read_access` for the same reason: the video twin used to inline a narrower rule
    that recognized only Shared and Public visibility, so a video on a board explicitly shared with
    you was unreadable while the image beside it was fine. Two media kinds on one board should not
    disagree about who can see them.
    """
    if current_user.is_admin:
        return

    owner = ApiDependencies.invoker.services.video_records.get_user_id(video_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_video_records.get_board_for_video(video_name)
    if board_id is not None:
        assert_board_read_access(board_id, current_user)
        return

    raise HTTPException(status_code=403, detail="Not authorized to access this video")


def assert_video_owner(video_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not mutate the video.

    The mutation twin of `assert_image_owner`: the direct owner, the owner of the board it sits on,
    or a Public board, which grants contribution rights.
    """
    if current_user.is_admin:
        return

    owner = ApiDependencies.invoker.services.video_records.get_user_id(video_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_video_records.get_board_for_video(video_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.user_id == current_user.user_id:
                return
            if board.board_visibility == BoardVisibility.Public:
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to modify this video")


def assert_board_read_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not read images from this board.

    Access is granted when ANY of these hold:
    - The user is an admin.
    - The user owns the board.
    - The board visibility is Shared or Public.
    - The board is explicitly shared with the user.
    """
    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")

    if current_user.is_admin:
        return

    if board.user_id == current_user.user_id:
        return

    if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
        return

    if ApiDependencies.invoker.services.board_records.is_board_shared_with_user(board_id, current_user.user_id):
        return

    raise HTTPException(status_code=403, detail="Not authorized to access this board")
