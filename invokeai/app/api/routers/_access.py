"""Cross-router authorization helpers.

These helpers are imported by multiple router modules. Keep them free of router
specifics so any route can call them after resolving `current_user`.
"""

from fastapi import HTTPException

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_records.board_records_common import (
    BoardRecordNotFoundException,
    BoardVisibility,
)


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
    owner = ApiDependencies.invoker.services.image_records.get_user_id(image_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_image_records.get_board_for_image(image_name)
    if board_id is not None:
        # The board *record*, not its DTO: the decision needs only the owner and the
        # visibility, and the DTO would drag in cover-image resolution plus three COUNT
        # aggregates — five extra queries and five extra ways to fail per name.
        #
        # Only a board positively known to be gone falls through to the 403. A storage error
        # propagates instead of being caught here: `board_records.get` deliberately does not
        # translate sqlite errors into not-found, and a caller that cannot decide ownership
        # must not report the name as an ordinary permission denial — the batch loops treat a
        # 403 as a silent auth skip, which turned a locked database into images dropped from
        # the response with no failure reported at all.
        try:
            board = ApiDependencies.invoker.services.board_records.get(board_id)
        except BoardRecordNotFoundException:
            pass
        else:
            if board.user_id == current_user.user_id:
                return
            if board.board_visibility == BoardVisibility.Public:
                return

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
