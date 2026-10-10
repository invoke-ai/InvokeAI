"""Cross-router authorization helpers.

These helpers are imported by multiple router modules. Keep them free of router
specifics so any route can call them after resolving `current_user`.
"""

from collections.abc import Iterable

from fastapi import HTTPException

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_access.board_access_common import (
    BoardAction,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.project_records.project_records_common import ProjectRecordNotFoundError


def _require(
    board_id: str, current_user: CurrentUserOrDefault, actions: Iterable[BoardAction], inbox_left_to_storage: bool
) -> None:
    """Raise unless the caller may take every action on the board. A board the caller cannot read answers 404, as
    a missing one does, so board ids reveal nothing; one they can read but not change answers 403."""
    try:
        access = ApiDependencies.invoker.services.board_access.get(
            board_id, current_user.user_id, current_user.is_admin
        )
    except BoardRecordNotFoundException:
        raise HTTPException(status_code=404, detail="Board not found")
    if not access.allows(BoardAction.Read):
        raise HTTPException(status_code=404, detail="Board not found")
    allows = access.role_allows if inbox_left_to_storage else access.allows
    if not all(allows(action) for action in actions):
        raise HTTPException(status_code=403, detail="Not authorized to change this board")


def assert_board_changes(board_id: str, current_user: CurrentUserOrDefault, actions: Iterable[BoardAction]) -> None:
    """Raise unless the caller's role allows every change. What a project's inbox refuses to everyone is left to
    the storage, which answers it with a 409 naming the project as the place to make the change."""
    _require(board_id, current_user, actions, inbox_left_to_storage=True)


def assert_board_read_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise unless the caller may read the board."""
    _require(board_id, current_user, [BoardAction.Read], inbox_left_to_storage=False)


def assert_board_write_access(board_id: str | None, current_user: CurrentUserOrDefault) -> None:
    """Raise unless the caller may put media on this board. `None` means "no board" and is always allowed, so
    upload routes can pass their optional board straight through."""
    if board_id is not None:
        _require(board_id, current_user, [BoardAction.AddMedia], inbox_left_to_storage=False)


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


def assert_image_action(image_name: str, current_user: CurrentUserOrDefault, action: ItemAction) -> None:
    """Raise 403 unless the caller may do the action to the image, or 404 when the image is gone.

    The refusal stays 403 for an image that exists: a client drops its references to an image it is told is gone,
    and a refusal can be reversed (see `_assert_image_record_exists`).
    """
    if current_user.is_admin:
        return
    try:
        ApiDependencies.invoker.services.board_access.require_image(
            image_name, current_user.user_id, current_user.is_admin, action
        )
    except ItemAccessDeniedError:
        _assert_image_record_exists(image_name)
        raise HTTPException(status_code=403, detail="Not authorized to access this image")


def assert_image_read_access(image_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise unless the caller may view the image: their own, or on a board they may read."""
    assert_image_action(image_name, current_user, ItemAction.Read)


def assert_video_action(video_name: str, current_user: CurrentUserOrDefault, action: ItemAction) -> None:
    """The video twin of `assert_image_action`: two media kinds on one board follow one rule."""
    if current_user.is_admin:
        return
    try:
        ApiDependencies.invoker.services.board_access.require_video(
            video_name, current_user.user_id, current_user.is_admin, action
        )
    except ItemAccessDeniedError:
        if not ApiDependencies.invoker.services.video_records.exists(video_name):
            raise HTTPException(status_code=404, detail="Video not found")
        raise HTTPException(status_code=403, detail="Not authorized to access this video")


def assert_video_read_access(video_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise unless the caller may view the video: their own, or on a board they may read."""
    assert_video_action(video_name, current_user, ItemAction.Read)


def item_action_for_changes(changes: object) -> ItemAction:
    """Starring someone else's item is the board's managers' to do; any other change to its record is its
    owner's."""
    fields = getattr(changes, "model_fields_set", set())
    return ItemAction.Star if fields and fields <= {"starred"} else ItemAction.Edit


def assert_project_owned(project_id: str | None, current_user: CurrentUserOrDefault) -> None:
    """Raise 404 unless `project_id` names one of the caller's own projects.

    Provenance is recorded only from a validated identity: an admin cannot attribute media to
    someone else's project, and a project id nobody owns reads as absent rather than as a
    permission decision, matching how the queue treats `Batch.project_id`.
    """
    if project_id is None:
        return
    try:
        ApiDependencies.invoker.services.project_records.get_board_id(current_user.user_id, project_id)
    except ProjectRecordNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
