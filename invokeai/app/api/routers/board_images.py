from enum import Enum, auto

from fastapi import Body, HTTPException
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.api.routers.images import MAX_IMAGE_BATCH_SIZE, ImageName
from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException
from invokeai.app.services.images.images_common import AddImagesToBoardResult, RemoveImagesFromBoardResult

board_images_router = APIRouter(prefix="/v1/board_images", tags=["boards"])


def _assert_board_write_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not mutate the given board.

    Write access is granted when ANY of these hold:
    - The user is an admin.
    - The user owns the board.
    - The board visibility is Public (public boards accept contributions from any user).

    Reads the board *record*, not its DTO. The decision needs only the owner and the
    visibility, while BoardService.get_dto also resolves the cover image and runs three COUNT
    aggregates over the board's contents — six queries to answer a question two columns settle.
    That cost is the only reason a batch route would be tempted to decide once and reuse the
    answer for every name, and reusing it is what lets a permission revoked mid-batch keep
    working until the request ends. One indexed SELECT per name is cheap enough to re-decide.
    (These routes are sync `def`, so the queries occupy a threadpool worker rather than the
    event loop — but a 1000-name batch still holds one for six thousand round trips.)
    """
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    try:
        board = ApiDependencies.invoker.services.board_records.get(board_id)
    except BoardRecordNotFoundException:
        raise HTTPException(status_code=404, detail="Board not found")
    # Anything else — a locked or unreadable database — propagates. Catching it here would
    # answer "no such board", which the batch loops below treat as a name to skip: a disk error
    # would then drop names out of the response entirely, reported neither as moved nor as
    # failed, and the client would show the move as done until the next refresh.
    if current_user.is_admin:
        return
    if board.user_id == current_user.user_id:
        return
    if board.board_visibility == BoardVisibility.Public:
        return
    raise HTTPException(status_code=403, detail="Not authorized to modify this board")


def _image_record_exists(image_name: str) -> bool:
    """True if the image record is still present, False if it has been deleted.

    A storage error answers True: only a record positively known to be gone may be downgraded
    from a reported failure to a silent skip. `ImageRecordStorage.get` no longer translates
    sqlite errors into not-found, so the two cases are distinguishable here.
    """
    try:
        ApiDependencies.invoker.services.image_records.get(image_name)
        return True
    except ImageRecordNotFoundException:
        return False
    except Exception:
        return True


class _ScopedRemoveOutcome(Enum):
    """What a scoped board-image DELETE turned out to have done, judged by its row count."""

    REMOVED = auto()
    """The row was deleted -- or the image was concurrently uncategorized by someone else, in
    which case the postcondition the caller asked for (off every board) holds, and reporting it
    removed is what lets the client's stale view of the old board catch up. Safe to report,
    unlike a deleted name: the DTO exists, so the tag-driven refetches succeed."""

    MOVED = auto()
    """Now on another board: the ask is not satisfied, and a retry will re-read and
    re-authorize against the board the image actually sits on now. Report as failed."""

    GONE = auto()
    """Image deleted concurrently: a skip, never a success -- reporting it removed would drive
    the client's tag-driven getImageDTO refetch straight into a 404."""


def _remove_from_board_and_classify(image_name: str, old_board_id: str) -> _ScopedRemoveOutcome:
    """Runs the scoped DELETE for a name read as sitting on `old_board_id`, then classifies.

    The scoped DELETE misses when the image leaves `old_board_id` between the caller's read
    and this write. The row count is the only signal the scope held: ignore it and the route
    reports a removal that did not happen, invalidating the wrong boards while the client
    counts the name as done. A zero-row miss is classified by where the image is now.

    The existence probe is direct rather than through `_image_record_exists`: that helper
    answers True on a storage error, which is the conservative bias where True means "report
    as failed" (the add loop) -- here True means "report as removed", and a transient storage
    error must not manufacture a success. Storage errors -- the DELETE's own and the
    classification reads' -- propagate instead: a name whose state cannot be decided must be
    reported by the caller as failed, never as done.

    `old_board_id` must be a real board id: uncategorized is the absence of a row, so a scoped
    DELETE for "none" cannot match and the classification would spend two reads confirming
    what the caller's DTO read already said.
    """
    deleted_rows = ApiDependencies.invoker.services.board_images.remove_image_from_board(
        image_name=image_name, board_id=old_board_id
    )
    if deleted_rows > 0:
        return _ScopedRemoveOutcome.REMOVED
    current_board_id = ApiDependencies.invoker.services.board_image_records.get_board_for_image(image_name)
    if current_board_id is not None:
        return _ScopedRemoveOutcome.MOVED
    try:
        ApiDependencies.invoker.services.image_records.get(image_name)
    except ImageRecordNotFoundException:
        return _ScopedRemoveOutcome.GONE
    return _ScopedRemoveOutcome.REMOVED


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
        failed_images: set[str] = set()
        affected_boards: set[str] = set()
        if old_board_id == "none":
            # Already off every board — the postcondition holds without a write. No
            # board_images row ever carries board_id="none", so a scoped DELETE could not
            # match anyway; see the identical shortcut in the batch loop below.
            removed_images.add(image_name)
            affected_boards.add("none")
        else:
            # The same row-count classification the batch loop uses. This route used to
            # ignore the count, so an image that left old_board_id between the read above and
            # the write was reported removed anyway — a false success that invalidated the
            # wrong boards and told the client the name was done.
            outcome = _remove_from_board_and_classify(image_name, old_board_id)
            if outcome is _ScopedRemoveOutcome.REMOVED:
                removed_images.add(image_name)
                affected_boards.add("none")
                affected_boards.add(old_board_id)
            elif outcome is _ScopedRemoveOutcome.MOVED:
                failed_images.add(image_name)
            # GONE lands in neither list, matching the batch route's treatment of a name that
            # vanished mid-flight; the client's refetches surface the deletion.
        return RemoveImagesFromBoardResult(
            removed_images=list(removed_images),
            failed_images=list(failed_images),
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
            # The destination decision sits in its own arm because its refusal means the
            # opposite of the per-image one. A foreign or vanished *image* is that name's own
            # problem — a skip, matching the other batch routes. A revoked or deleted
            # *destination* is the whole request's problem: every remaining name meets it too,
            # and treating those as skips answers 201 with empty lists, which the client reads
            # as success and clears the user's selection over. Still re-decided per name, and
            # the loop keeps going rather than aborting: access restored mid-batch lets later
            # names land, and every name refused while it was gone is reported as failed.
            try:
                # Re-decided per name rather than resting on the check above. Write access to
                # the target board can be revoked while the batch is running — a board flipped
                # from Public to Private — and a decision taken once at the top of a 1000-name
                # request would let a contributor keep writing to it for the rest of the batch.
                _assert_board_write_access(board_id, current_user)
            except HTTPException:
                failed_images.add(image_name)
                continue
            except Exception:
                # The helper propagates storage errors precisely so they are not mistaken for
                # "no such board"; a name whose destination could not be decided is reported.
                failed_images.add(image_name)
                continue
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
                #
                # Except that a name deleted between the ownership check and the insert lands
                # here too, and not as something recognizable: board_images.image_name is a
                # foreign key onto images.image_name, so the INSERT fails with a bare
                # sqlite3.IntegrityError. Nothing in the exception says "gone", so the record
                # is probed instead — only on this path, so the happy path pays nothing.
                if not _image_record_exists(image_name):
                    continue
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

            # The one authorization decision, and it is taken fresh for every name. Memoizing
            # it per board is tempting — the skip removed the early abort that used to cap an
            # unauthorized batch at one check — but a cached True outlives the permission it
            # recorded: flip a board from Public to Private mid-batch and the rest of the names
            # are removed on an answer that is no longer true. _assert_board_write_access reads
            # the board record (one indexed SELECT), so re-deciding costs about what the
            # get_dto() it replaced cost once.
            #
            # Kept out of the try below so that the only way to skip a name for auth is this
            # branch. Note the two failure modes are not the same: "not allowed" is a skip,
            # while a storage error means the decision could not be taken at all, and a name we
            # could not decide about must be reported rather than silently dropped.
            if old_board_id != "none":
                try:
                    _assert_board_write_access(old_board_id, current_user)
                except HTTPException:
                    continue
                except Exception:
                    failed_images.add(image_name)
                    continue

            # No board_images row ever carries board_id="none" — uncategorized is the absence
            # of a row — so for a name already off every board the scoped DELETE cannot match
            # and the zero-row classification below would spend two reads confirming what the
            # DTO already said. Same report the classification would produce, minus the reads.
            if old_board_id == "none":
                removed_images.add(image_name)
                affected_boards.add("none")
                continue

            try:
                outcome = _remove_from_board_and_classify(image_name, old_board_id)
                if outcome is _ScopedRemoveOutcome.REMOVED:
                    removed_images.add(image_name)
                    affected_boards.add("none")
                    affected_boards.add(old_board_id)
                elif outcome is _ScopedRemoveOutcome.MOVED:
                    failed_images.add(image_name)
                # GONE: a skip, exactly as the gone-block above treats a name that vanished
                # before the loop reached it.
            except Exception:
                # A genuine storage failure, not an auth/404 skip — see add_images_to_board.
                # The zero-row classification's own reads land here too: a name whose state
                # cannot be decided is reported, never dropped.
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
