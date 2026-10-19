"""Cross-router authorization helpers.

These helpers are imported by multiple router modules. Keep them free of router
specifics so any route can call them after resolving `current_user`.
"""

from collections.abc import Iterable
from typing import Optional, Protocol

from fastapi import HTTPException

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.board_access.board_access_common import (
    BoardAccess,
    BoardAction,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.project_records.project_records_common import ProjectRecordNotFoundError


def _require(
    board_id: str, current_user: CurrentUserOrDefault, actions: Iterable[BoardAction], inbox_left_to_storage: bool
) -> BoardAccess:
    """The caller's access to the board, when it allows every action; raises otherwise.

    Only a missing board answers 404. A board the caller may not read answers 403, like an image or a video they
    may not read (see `_assert_image_record_exists`), because the two refusals mean opposite things to a client:
    gone is permanent, and the client drops its selection, auto-add target and cached items for the board; denied
    is a permission decision an owner can reverse by flipping the board back to shared or re-granting membership,
    and nothing would restore what the client dropped over it. Board ids are generated UUIDs, so learning that
    one exists buys an attacker nothing they could enumerate. A board the caller can read but not change
    answers 403 too, worded as the change refused. Each 403 carries a `code` beside its message: a client that
    lost access drops nothing, while one whose role is stale only refreshes the board.
    """
    try:
        access = ApiDependencies.invoker.services.board_access.get(
            board_id, current_user.user_id, current_user.is_admin
        )
    except BoardRecordNotFoundException:
        raise HTTPException(status_code=404, detail="Board not found")
    if not access.allows(BoardAction.Read):
        raise HTTPException(
            status_code=403, detail={"code": "board_unreadable", "message": "Not authorized to access this board"}
        )
    allows = access.role_allows if inbox_left_to_storage else access.allows
    if not all(allows(action) for action in actions):
        raise HTTPException(
            status_code=403,
            detail={"code": "board_action_denied", "message": "Not authorized to change this board"},
        )
    return access


def assert_board_changes(
    board_id: str, current_user: CurrentUserOrDefault, actions: Iterable[BoardAction]
) -> BoardAccess:
    """The caller's access, when their role allows every change; raises otherwise. What a project's inbox refuses
    to everyone is left to the storage, which answers it with a 409 naming the project as the place to make the
    change. The access is returned so a route can tell an owner acting as one from an admin overriding."""
    return _require(board_id, current_user, actions, inbox_left_to_storage=True)


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


class _MediaDTO(Protocol):
    """What the redaction needs of an image or video DTO."""

    project_id: Optional[str]


def redact_project_ids(dtos: Iterable[_MediaDTO], current_user: CurrentUserOrDefault) -> None:
    """Blank `project_id` on every DTO whose project is not the caller's own, in place and in one query.

    Provenance is only ever recorded from the uploader's own project (`assert_project_owned`), so a viewer of a
    shared board has no business learning it: a project id confirmed here, combined with the 409 on creating a
    project with an existing id, would tell them the project exists. Who sees which project is the access
    service's decision (`visible_project_ids`); this only applies it to the response.
    """
    attributed = [dto for dto in dtos if dto.project_id is not None]
    if not attributed:
        return
    visible = ApiDependencies.invoker.services.board_access.visible_project_ids(
        {dto.project_id for dto in attributed if dto.project_id is not None},
        current_user.user_id,
        current_user.is_admin,
    )
    for dto in attributed:
        if dto.project_id not in visible:
            dto.project_id = None
