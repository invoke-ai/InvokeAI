from fastapi import Body, HTTPException
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.api.routers.images import MAX_IMAGE_BATCH_SIZE, ImageName
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException
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
def add_image_to_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Body(description="The id of the board to add to"),
    image_name: str = Body(description="The name of the image to add"),
) -> AddImagesToBoardResult:
    """Creates a board_image"""
    _assert_board_write_access(board_id, current_user)
    _assert_image_direct_owner(image_name, current_user)
    assert_image_move_maintenance_inactive()
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
            # Single-image route: a failure here is a 500, never a partial success.
            failed_images=[],
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
def remove_image_from_board(
    current_user: CurrentUserOrDefault,
    image_name: str = Body(description="The name of the image to remove", embed=True),
) -> RemoveImagesFromBoardResult:
    """Removes an image from its board, if it had one"""
    try:
        old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
        if old_board_id != "none":
            _assert_board_write_access(old_board_id, current_user)
        assert_image_move_maintenance_inactive()
        removed_images: set[str] = set()
        affected_boards: set[str] = set()
        ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
        removed_images.add(image_name)
        affected_boards.add("none")
        affected_boards.add(old_board_id)
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            # Single-image route: a failure here is a 500, never a partial success.
            failed_images=[],
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
def add_images_to_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Body(description="The id of the board to add to"),
    image_names: list[ImageName] = Body(
        description="The names of the images to add", embed=True, max_length=MAX_IMAGE_BATCH_SIZE
    ),
) -> AddImagesToBoardResult:
    """Adds a list of images to a board"""
    _assert_board_write_access(board_id, current_user)
    try:
        assert_image_move_maintenance_inactive()
    except HTTPException:
        for image_name in image_names:
            _assert_image_direct_owner(image_name, current_user)
        raise

    try:
        # Skip — but do not re-raise — auth failures so a foreign name mid-batch doesn't
        # discard the response payload for images that were already moved. Re-raising turned
        # partial successes into an error-shaped response, so the client never invalidated
        # caches for the images that did move and the UI kept showing them on their old board
        # until the next full refresh. Matches star_images_in_list and delete_images_from_list.
        added_images: set[str] = set()
        failed_images: set[str] = set()
        affected_boards: set[str] = set()
        # Dedup while preserving order — a repeated name would otherwise be processed twice
        # and could land in both added_images and failed_images.
        for image_name in dict.fromkeys(image_names):
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
                continue
            except Exception:
                # A genuine storage failure, not an auth/404 skip: it used to be swallowed by
                # `pass`, so the client counted the image as moved and the move silently
                # reverted on reload.
                failed_images.add(image_name)
        return AddImagesToBoardResult(
            added_images=list(added_images),
            failed_images=list(failed_images),
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
def remove_images_from_board(
    current_user: CurrentUserOrDefault,
    image_names: list[ImageName] = Body(
        description="The names of the images to remove", embed=True, max_length=MAX_IMAGE_BATCH_SIZE
    ),
) -> RemoveImagesFromBoardResult:
    """Removes a list of images from their board, if they had one"""
    try:
        assert_image_move_maintenance_inactive()
    except HTTPException:
        for image_name in image_names:
            try:
                old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
            except ImageRecordNotFoundException:
                # A name deleted by a concurrent session. The main loop treats that as a skip;
                # letting it escape from inside this handler would replace the 409 with a 500.
                continue
            if old_board_id != "none":
                _assert_board_write_access(old_board_id, current_user)
        raise

    try:
        # Skip — but do not re-raise — auth failures, for the same reason as add_images_to_board
        # above: one name on a board the caller cannot write must not discard the payload for
        # the images already removed.
        removed_images: set[str] = set()
        failed_images: set[str] = set()
        affected_boards: set[str] = set()
        # Decided once per board rather than once per name. The skip removed the early abort
        # that used to cap an unauthorized batch at one check, and _assert_board_write_access
        # goes through boards.get_dto() -- six queries including three COUNT aggregates over
        # the board's contents. Unmemoized, a 1000-name batch on one board is 6000 synchronous
        # queries on the event loop, which is exactly what MAX_IMAGE_BATCH_SIZE exists to stop.
        board_is_writable: dict[str, bool] = {}

        def _may_write(board_id: str) -> bool:
            if board_id not in board_is_writable:
                try:
                    _assert_board_write_access(board_id, current_user)
                    board_is_writable[board_id] = True
                except HTTPException:
                    board_is_writable[board_id] = False
            return board_is_writable[board_id]

        # Dedup while preserving order — a repeated name would otherwise be processed twice
        # and could land in both removed_images and failed_images.
        for image_name in dict.fromkeys(image_names):
            try:
                old_board_id = ApiDependencies.invoker.services.images.get_dto(image_name).board_id or "none"
            except ImageRecordNotFoundException:
                # The image is gone — deleted by a concurrent session between the client
                # building its selection and this request. That is a skip, not a failure, and
                # must not be toasted as one. Resolved in its own block because unlike the
                # other routes this one reads the DTO *before* any authorization check, so a
                # 404 here would otherwise be indistinguishable from a storage failure below.
                # Narrow on purpose: a real storage error must still reach failed_images.
                continue
            except Exception:
                failed_images.add(image_name)
                continue

            # The one authorization decision, outside the try below so that the only way to
            # skip a name for auth is this branch. Folding it into the try would leave two
            # paths to the same outcome, and neither would be individually load-bearing.
            if old_board_id != "none" and not _may_write(old_board_id):
                continue

            try:
                ApiDependencies.invoker.services.board_images.remove_image_from_board(image_name=image_name)
                removed_images.add(image_name)
                affected_boards.add("none")
                affected_boards.add(old_board_id)
            except Exception:
                # A genuine storage failure, not an auth/404 skip — see add_images_to_board.
                failed_images.add(image_name)
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            failed_images=list(failed_images),
            affected_boards=list(affected_boards),
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove images from board")
