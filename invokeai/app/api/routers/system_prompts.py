from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Path

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptChanges,
    SystemPromptNotFoundError,
    SystemPromptRecordDTO,
    SystemPromptWithoutId,
)

system_prompts_router = APIRouter(prefix="/v1/system_prompts", tags=["system_prompts"])


@system_prompts_router.get(
    "/",
    operation_id="list_system_prompts",
    responses={200: {"model": list[SystemPromptRecordDTO]}},
)
async def list_system_prompts(current_user: CurrentUserOrDefault) -> list[SystemPromptRecordDTO]:
    """Lists system prompts visible to the current user (own + public)."""
    config = ApiDependencies.invoker.services.configuration
    # Admins (and single-user installs) see everything; multiuser non-admins are scoped to own + public.
    user_id_filter: Optional[str] = None
    if config.multiuser and not current_user.is_admin:
        user_id_filter = current_user.user_id
    return ApiDependencies.invoker.services.system_prompt_records.get_many(user_id=user_id_filter)


@system_prompts_router.get(
    "/i/{system_prompt_id}",
    operation_id="get_system_prompt",
    responses={200: {"model": SystemPromptRecordDTO}},
)
async def get_system_prompt(
    current_user: CurrentUserOrDefault,
    system_prompt_id: str = Path(description="The id of the system prompt to get"),
) -> SystemPromptRecordDTO:
    """Gets a system prompt by id."""
    try:
        prompt = ApiDependencies.invoker.services.system_prompt_records.get(system_prompt_id)
    except SystemPromptNotFoundError:
        raise HTTPException(status_code=404, detail="System prompt not found")

    config = ApiDependencies.invoker.services.configuration
    if config.multiuser:
        is_owner = prompt.user_id == current_user.user_id
        if not (is_owner or prompt.is_public or current_user.is_admin):
            raise HTTPException(status_code=403, detail="Not authorized to access this system prompt")
    return prompt


@system_prompts_router.post(
    "/",
    operation_id="create_system_prompt",
    responses={200: {"model": SystemPromptRecordDTO}},
)
async def create_system_prompt(
    current_user: CurrentUserOrDefault,
    system_prompt: SystemPromptWithoutId = Body(description="The system prompt to create"),
) -> SystemPromptRecordDTO:
    """Creates a new system prompt owned by the current user."""
    # Single-user: shared so legacy/single-user behaviour is unchanged. Multiuser: private by default.
    config = ApiDependencies.invoker.services.configuration
    is_public = not config.multiuser
    return ApiDependencies.invoker.services.system_prompt_records.create(
        system_prompt, user_id=current_user.user_id, is_public=is_public
    )


@system_prompts_router.patch(
    "/i/{system_prompt_id}",
    operation_id="update_system_prompt",
    responses={200: {"model": SystemPromptRecordDTO}},
)
async def update_system_prompt(
    current_user: CurrentUserOrDefault,
    system_prompt_id: str = Path(description="The id of the system prompt to update"),
    changes: SystemPromptChanges = Body(description="The changes to apply"),
) -> SystemPromptRecordDTO:
    """Updates a system prompt. Only the owner or an admin may update."""
    config = ApiDependencies.invoker.services.configuration
    if config.multiuser:
        try:
            existing = ApiDependencies.invoker.services.system_prompt_records.get(system_prompt_id)
        except SystemPromptNotFoundError:
            raise HTTPException(status_code=404, detail="System prompt not found")
        if not current_user.is_admin and existing.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to update this system prompt")
    user_id = None if current_user.is_admin else current_user.user_id
    try:
        return ApiDependencies.invoker.services.system_prompt_records.update(
            system_prompt_id, changes, user_id=user_id
        )
    except SystemPromptNotFoundError:
        raise HTTPException(status_code=404, detail="System prompt not found")


@system_prompts_router.delete(
    "/i/{system_prompt_id}",
    operation_id="delete_system_prompt",
)
async def delete_system_prompt(
    current_user: CurrentUserOrDefault,
    system_prompt_id: str = Path(description="The id of the system prompt to delete"),
) -> None:
    """Deletes a system prompt. Only the owner or an admin may delete."""
    config = ApiDependencies.invoker.services.configuration
    if config.multiuser:
        try:
            existing = ApiDependencies.invoker.services.system_prompt_records.get(system_prompt_id)
        except SystemPromptNotFoundError:
            raise HTTPException(status_code=404, detail="System prompt not found")
        if not current_user.is_admin and existing.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this system prompt")
    user_id = None if current_user.is_admin else current_user.user_id
    ApiDependencies.invoker.services.system_prompt_records.delete(system_prompt_id, user_id=user_id)
