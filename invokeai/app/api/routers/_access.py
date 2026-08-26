"""Cross-router authorization helpers.

These helpers are imported by multiple router modules. Keep them free of router
specifics so any route can call them after resolving `current_user`.
"""

from fastapi import HTTPException

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_records.board_records_common import (
    BoardRecord,
    BoardRecordNotFoundException,
    BoardVisibility,
)


def _get_board_record(board_id: str) -> BoardRecord:
    """Get a board record, translating only a confirmed missing board to 404."""
    try:
        return ApiDependencies.invoker.services.board_records.get(board_id)
    except BoardRecordNotFoundException:
        raise HTTPException(status_code=404, detail="Board not found")


def _board_grants_contribution(board_id: str, current_user: CurrentUserOrDefault) -> bool:
    """Whether this user may put media on the board: they own it, or it is Public.

    Shared boards are deliberately excluded — `Shared` grants visibility, not contribution.
    Storage errors propagate so a batch cannot silently turn an undecidable permission check into
    an ordinary denial.
    """
    try:
        board = ApiDependencies.invoker.services.board_records.get(board_id)
    except BoardRecordNotFoundException:
        return False

    return board.user_id == current_user.user_id or board.board_visibility == BoardVisibility.Public


def _board_grants_read_access(board_id: str, current_user: CurrentUserOrDefault) -> bool:
    """Whether the user may read a board, returning False only when it is confirmed missing or denied."""
    try:
        board = ApiDependencies.invoker.services.board_records.get(board_id)
    except BoardRecordNotFoundException:
        return False

    if board.user_id == current_user.user_id:
        return True
    if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
        return True
    return ApiDependencies.invoker.services.board_records.is_board_shared_with_user(board_id, current_user.user_id)


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
    if board_id is not None and _board_grants_contribution(board_id, current_user):
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

    A storage error propagates rather than answering either, so an unreadable database cannot
    present as a deleted image and take the user's references down with it. `exists` is a bare
    row probe rather than `get` for the same reason from the other side: `get` deserializes, so
    a row written by a newer version — an enum value this one does not know — would fail exactly
    as absence does, and a live image would be reported gone.

    The cost is that an authenticated caller can now tell an absent image from one they may not
    read. Image names are generated UUIDs, so this buys an attacker nothing they could enumerate,
    and it is the answer admins have always received.
    """
    if not ApiDependencies.invoker.services.image_records.exists(image_name):
        raise HTTPException(status_code=404, detail="Image not found")


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
    if board_id is not None and _board_grants_read_access(board_id, current_user):
        return

    _assert_image_record_exists(image_name)
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

    board = _get_board_record(board_id)

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
    if board_id is not None and _board_grants_read_access(board_id, current_user):
        return

    if not ApiDependencies.invoker.services.video_records.exists(video_name):
        raise HTTPException(status_code=404, detail="Video not found")
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
    if board_id is not None and _board_grants_contribution(board_id, current_user):
        return

    raise HTTPException(status_code=403, detail="Not authorized to modify this video")


def assert_board_read_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not read images from this board.

    Access is granted when ANY of these hold:
    - The user is an admin.
    - The user owns the board.
    - The board visibility is Shared or Public.
    - The board is explicitly shared with the user.
    """
    board = _get_board_record(board_id)

    if current_user.is_admin:
        return

    if board.user_id == current_user.user_id:
        return

    if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
        return

    if ApiDependencies.invoker.services.board_records.is_board_shared_with_user(board_id, current_user.user_id):
        return

    raise HTTPException(status_code=403, detail="Not authorized to access this board")
