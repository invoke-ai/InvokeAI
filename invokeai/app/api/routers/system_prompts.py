from fastapi import APIRouter, Body, HTTPException, Path

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
async def list_system_prompts() -> list[SystemPromptRecordDTO]:
    """Lists all system prompts."""
    return ApiDependencies.invoker.services.system_prompt_records.get_many()


@system_prompts_router.get(
    "/i/{system_prompt_id}",
    operation_id="get_system_prompt",
    responses={200: {"model": SystemPromptRecordDTO}},
)
async def get_system_prompt(
    system_prompt_id: str = Path(description="The id of the system prompt to get"),
) -> SystemPromptRecordDTO:
    """Gets a system prompt by id."""
    try:
        return ApiDependencies.invoker.services.system_prompt_records.get(system_prompt_id)
    except SystemPromptNotFoundError:
        raise HTTPException(status_code=404, detail="System prompt not found")


@system_prompts_router.post(
    "/",
    operation_id="create_system_prompt",
    responses={200: {"model": SystemPromptRecordDTO}},
)
async def create_system_prompt(
    system_prompt: SystemPromptWithoutId = Body(description="The system prompt to create"),
) -> SystemPromptRecordDTO:
    """Creates a new system prompt."""
    return ApiDependencies.invoker.services.system_prompt_records.create(system_prompt)


@system_prompts_router.patch(
    "/i/{system_prompt_id}",
    operation_id="update_system_prompt",
    responses={200: {"model": SystemPromptRecordDTO}},
)
async def update_system_prompt(
    system_prompt_id: str = Path(description="The id of the system prompt to update"),
    changes: SystemPromptChanges = Body(description="The changes to apply"),
) -> SystemPromptRecordDTO:
    """Updates a system prompt."""
    try:
        return ApiDependencies.invoker.services.system_prompt_records.update(system_prompt_id, changes)
    except SystemPromptNotFoundError:
        raise HTTPException(status_code=404, detail="System prompt not found")


@system_prompts_router.delete(
    "/i/{system_prompt_id}",
    operation_id="delete_system_prompt",
)
async def delete_system_prompt(
    system_prompt_id: str = Path(description="The id of the system prompt to delete"),
) -> None:
    """Deletes a system prompt."""
    ApiDependencies.invoker.services.system_prompt_records.delete(system_prompt_id)
