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
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException


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


def _assert_image_record_exists(image_name: str) -> None:
    """Turn a refusal into a 404 when the image is positively gone.

    The two refusals mean opposite things to a client holding a reference to the image — a
    workflow's image field, a reference image on a canvas layer. Gone is permanent, and the
    reference should be dropped. Denied is a permission decision that can be reversed (a board
    flipped back to Shared, an owner re-granting access), and dropping the reference over one
    destroys work the user cannot get back by restoring the permission.

    Nothing above can tell them apart: the ownership test rests on `images.user_id`, which is
    gone with the row, so a deleted image reaches that same 403 as a foreign one. So the
    distinction is made here, on the refusal path only — the happy path pays nothing for it.

    A storage error propagates rather than answering either. `image_records.get` deliberately
    does not translate sqlite errors into not-found, so an unreadable database cannot present as
    a deleted image and take the user's references down with it.

    The cost is that an authenticated caller can now tell an absent image from one they may not
    read. Image names are generated UUIDs, so this buys an attacker nothing they could enumerate,
    and it is the answer admins have always received.
    """
    try:
        ApiDependencies.invoker.services.image_records.get(image_name)
    except ImageRecordNotFoundException:
        raise HTTPException(status_code=404, detail="Image not found") from None


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
        # See `assert_image_owner` for why this reads the board record and catches only
        # not-found: a lookup that cannot be decided must not present as a permission decision.
        try:
            board = ApiDependencies.invoker.services.board_records.get(board_id)
        except BoardRecordNotFoundException:
            pass
        else:
            if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
                return

    _assert_image_record_exists(image_name)
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
