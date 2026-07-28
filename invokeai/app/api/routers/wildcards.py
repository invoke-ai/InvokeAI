from fastapi import Body, HTTPException, Path
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardRecordDTO,
    WildcardWithoutId,
)

wildcards_router = APIRouter(prefix="/v1/wildcards", tags=["wildcards"])


def _get_owned_wildcard(wildcard_id: str, user_id: str) -> WildcardRecordDTO:
    """Loads a wildcard, treating someone else's as absent rather than forbidden.

    Answering 403 would confirm the id exists, so a non-owner gets the same 404 as a bad id.
    """
    try:
        wildcard = ApiDependencies.invoker.services.wildcard_records.get(wildcard_id)
    except WildcardNotFoundError:
        raise HTTPException(status_code=404, detail="Wildcard not found")

    if wildcard.user_id != user_id:
        raise HTTPException(status_code=404, detail="Wildcard not found")

    return wildcard


@wildcards_router.get(
    "/",
    operation_id="list_wildcards",
    responses={200: {"model": list[WildcardRecordDTO]}},
)
async def list_wildcards(current_user: CurrentUserOrDefault) -> list[WildcardRecordDTO]:
    """Lists the current user's wildcards."""
    return ApiDependencies.invoker.services.wildcard_records.get_many(user_id=current_user.user_id)


@wildcards_router.post(
    "/",
    operation_id="create_wildcard",
    responses={201: {"model": WildcardRecordDTO}},
    status_code=201,
)
async def create_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard: WildcardWithoutId = Body(description="The wildcard to create"),
) -> WildcardRecordDTO:
    """Creates a wildcard owned by the current user."""
    try:
        return ApiDependencies.invoker.services.wildcard_records.create(wildcard=wildcard, user_id=current_user.user_id)
    except WildcardNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@wildcards_router.patch(
    "/{wildcard_id}",
    operation_id="update_wildcard",
    responses={200: {"model": WildcardRecordDTO}},
)
async def update_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard_id: str = Path(description="The id of the wildcard to update"),
    changes: WildcardChanges = Body(description="The changes to apply"),
) -> WildcardRecordDTO:
    """Updates a wildcard owned by the current user."""
    _get_owned_wildcard(wildcard_id, current_user.user_id)

    try:
        return ApiDependencies.invoker.services.wildcard_records.update(wildcard_id=wildcard_id, changes=changes)
    except WildcardNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@wildcards_router.delete(
    "/{wildcard_id}",
    operation_id="delete_wildcard",
    status_code=204,
)
async def delete_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard_id: str = Path(description="The id of the wildcard to delete"),
) -> None:
    """Deletes a wildcard owned by the current user."""
    _get_owned_wildcard(wildcard_id, current_user.user_id)
    ApiDependencies.invoker.services.wildcard_records.delete(wildcard_id)
