from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.backend.util.logging import logging

client_state_router = APIRouter(prefix="/v1/client_state", tags=["client_state"])


@client_state_router.get(
    "/{queue_id}/get_by_key",
    operation_id="get_client_state_by_key",
    response_model=str | None,
)
async def get_client_state_by_key(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id (ignored, kept for backwards compatibility)"),
    key: str = Query(..., description="Key to get"),
) -> str | None:
    """Gets the client state for the current user (or system user if not authenticated)"""
    try:
        return ApiDependencies.invoker.services.client_state_persistence.get_by_key(current_user.user_id, key)
    except Exception as e:
        logging.error(f"Error getting client state: {e}")
        raise HTTPException(status_code=500, detail="Error getting client state")


@client_state_router.post(
    "/{queue_id}/set_by_key",
    operation_id="set_client_state",
    response_model=str,
)
async def set_client_state(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id (ignored, kept for backwards compatibility)"),
    key: str = Query(..., description="Key to set"),
    value: str = Body(..., description="Stringified value to set"),
) -> str:
    """Sets the client state for the current user (or system user if not authenticated)"""
    try:
        return ApiDependencies.invoker.services.client_state_persistence.set_by_key(current_user.user_id, key, value)
    except Exception as e:
        logging.error(f"Error setting client state: {e}")
        raise HTTPException(status_code=500, detail="Error setting client state")


@client_state_router.post(
    "/{queue_id}/delete",
    operation_id="delete_client_state",
    responses={204: {"description": "Client state deleted"}},
)
async def delete_client_state(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id (ignored, kept for backwards compatibility)"),
) -> None:
    """Deletes the client state for the current user (or system user if not authenticated)"""
    try:
        ApiDependencies.invoker.services.client_state_persistence.delete(current_user.user_id)
    except Exception as e:
        logging.error(f"Error deleting client state: {e}")
        raise HTTPException(status_code=500, detail="Error deleting client state")
