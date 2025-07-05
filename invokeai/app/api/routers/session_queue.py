from typing import Optional

from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    BatchStatus,
    CancelAllExceptCurrentResult,
    CancelByBatchIDsResult,
    CancelByDestinationResult,
    ClearResult,
    DeleteAllExceptCurrentResult,
    DeleteByDestinationResult,
    EnqueueBatchResult,
    FieldIdentifier,
    PruneResult,
    RetryItemsResult,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
    SessionQueueStatus,
)
from invokeai.app.services.shared.pagination import CursorPaginatedResults

session_queue_router = APIRouter(prefix="/v1/queue", tags=["queue"])


class SessionQueueAndProcessorStatus(BaseModel):
    """The overall status of session queue and processor"""

    queue: SessionQueueStatus
    processor: SessionProcessorStatus


class ValidationRunData(BaseModel):
    workflow_id: str = Field(description="The id of the workflow being published.")
    input_fields: list[FieldIdentifier] = Body(description="The input fields for the published workflow")
    output_fields: list[FieldIdentifier] = Body(description="The output fields for the published workflow")


@session_queue_router.post(
    "/{queue_id}/enqueue_batch",
    operation_id="enqueue_batch",
    responses={
        201: {"model": EnqueueBatchResult},
    },
)
async def enqueue_batch(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch: Batch = Body(description="Batch to process"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch in the queue"),
    validation_run_data: Optional[ValidationRunData] = Body(
        default=None,
        description="The validation run data to use for this batch. This is only used if this is a validation run.",
    ),
) -> EnqueueBatchResult:
    """Processes a batch and enqueues the output graphs for execution."""
    try:
        return await ApiDependencies.invoker.services.session_queue.enqueue_batch(
            queue_id=queue_id, batch=batch, prepend=prepend
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while enqueuing batch: {e}")


@session_queue_router.get(
    "/{queue_id}/list",
    operation_id="list_queue_items",
    responses={
        200: {"model": CursorPaginatedResults[SessionQueueItem]},
    },
)
async def list_queue_items(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    limit: int = Query(default=50, description="The number of items to fetch"),
    status: Optional[QUEUE_ITEM_STATUS] = Query(default=None, description="The status of items to fetch"),
    cursor: Optional[int] = Query(default=None, description="The pagination cursor"),
    priority: int = Query(default=0, description="The pagination cursor priority"),
    destination: Optional[str] = Query(default=None, description="The destination of queue items to fetch"),
) -> CursorPaginatedResults[SessionQueueItem]:
    """Gets cursor-paginated queue items"""

    try:
        return ApiDependencies.invoker.services.session_queue.list_queue_items(
            queue_id=queue_id,
            limit=limit,
            status=status,
            cursor=cursor,
            priority=priority,
            destination=destination,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while listing all items: {e}")


@session_queue_router.get(
    "/{queue_id}/list_all",
    operation_id="list_all_queue_items",
    responses={
        200: {"model": list[SessionQueueItem]},
    },
)
async def list_all_queue_items(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    destination: Optional[str] = Query(default=None, description="The destination of queue items to fetch"),
) -> list[SessionQueueItem]:
    """Gets all queue items"""
    try:
        return ApiDependencies.invoker.services.session_queue.list_all_queue_items(
            queue_id=queue_id,
            destination=destination,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while listing all queue items: {e}")


@session_queue_router.put(
    "/{queue_id}/processor/resume",
    operation_id="resume",
    responses={200: {"model": SessionProcessorStatus}},
)
async def resume(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionProcessorStatus:
    """Resumes session processor"""
    try:
        return ApiDependencies.invoker.services.session_processor.resume()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while resuming queue: {e}")


@session_queue_router.put(
    "/{queue_id}/processor/pause",
    operation_id="pause",
    responses={200: {"model": SessionProcessorStatus}},
)
async def Pause(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionProcessorStatus:
    """Pauses session processor"""
    try:
        return ApiDependencies.invoker.services.session_processor.pause()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while pausing queue: {e}")


@session_queue_router.put(
    "/{queue_id}/cancel_all_except_current",
    operation_id="cancel_all_except_current",
    responses={200: {"model": CancelAllExceptCurrentResult}},
)
async def cancel_all_except_current(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> CancelAllExceptCurrentResult:
    """Immediately cancels all queue items except in-processing items"""
    try:
        return ApiDependencies.invoker.services.session_queue.cancel_all_except_current(queue_id=queue_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling all except current: {e}")


@session_queue_router.put(
    "/{queue_id}/delete_all_except_current",
    operation_id="delete_all_except_current",
    responses={200: {"model": DeleteAllExceptCurrentResult}},
)
async def delete_all_except_current(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> DeleteAllExceptCurrentResult:
    """Immediately deletes all queue items except in-processing items"""
    try:
        return ApiDependencies.invoker.services.session_queue.delete_all_except_current(queue_id=queue_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while deleting all except current: {e}")


@session_queue_router.put(
    "/{queue_id}/cancel_by_batch_ids",
    operation_id="cancel_by_batch_ids",
    responses={200: {"model": CancelByBatchIDsResult}},
)
async def cancel_by_batch_ids(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch_ids: list[str] = Body(description="The list of batch_ids to cancel all queue items for", embed=True),
) -> CancelByBatchIDsResult:
    """Immediately cancels all queue items from the given batch ids"""
    try:
        return ApiDependencies.invoker.services.session_queue.cancel_by_batch_ids(
            queue_id=queue_id, batch_ids=batch_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling by batch id: {e}")


@session_queue_router.put(
    "/{queue_id}/cancel_by_destination",
    operation_id="cancel_by_destination",
    responses={200: {"model": CancelByDestinationResult}},
)
async def cancel_by_destination(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    destination: str = Query(description="The destination to cancel all queue items for"),
) -> CancelByDestinationResult:
    """Immediately cancels all queue items with the given origin"""
    try:
        return ApiDependencies.invoker.services.session_queue.cancel_by_destination(
            queue_id=queue_id, destination=destination
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling by destination: {e}")


@session_queue_router.put(
    "/{queue_id}/retry_items_by_id",
    operation_id="retry_items_by_id",
    responses={200: {"model": RetryItemsResult}},
)
async def retry_items_by_id(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_ids: list[int] = Body(description="The queue item ids to retry"),
) -> RetryItemsResult:
    """Immediately cancels all queue items with the given origin"""
    try:
        return ApiDependencies.invoker.services.session_queue.retry_items_by_id(queue_id=queue_id, item_ids=item_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while retrying queue items: {e}")


@session_queue_router.put(
    "/{queue_id}/clear",
    operation_id="clear",
    responses={
        200: {"model": ClearResult},
    },
)
async def clear(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> ClearResult:
    """Clears the queue entirely, immediately canceling the currently-executing session"""
    try:
        queue_item = ApiDependencies.invoker.services.session_queue.get_current(queue_id)
        if queue_item is not None:
            ApiDependencies.invoker.services.session_queue.cancel_queue_item(queue_item.item_id)
        clear_result = ApiDependencies.invoker.services.session_queue.clear(queue_id)
        return clear_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while clearing queue: {e}")


@session_queue_router.put(
    "/{queue_id}/prune",
    operation_id="prune",
    responses={
        200: {"model": PruneResult},
    },
)
async def prune(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> PruneResult:
    """Prunes all completed or errored queue items"""
    try:
        return ApiDependencies.invoker.services.session_queue.prune(queue_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while pruning queue: {e}")


@session_queue_router.get(
    "/{queue_id}/current",
    operation_id="get_current_queue_item",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
async def get_current_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> Optional[SessionQueueItem]:
    """Gets the currently execution queue item"""
    try:
        return ApiDependencies.invoker.services.session_queue.get_current(queue_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting current queue item: {e}")


@session_queue_router.get(
    "/{queue_id}/next",
    operation_id="get_next_queue_item",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
async def get_next_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> Optional[SessionQueueItem]:
    """Gets the next queue item, without executing it"""
    try:
        return ApiDependencies.invoker.services.session_queue.get_next(queue_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting next queue item: {e}")


@session_queue_router.get(
    "/{queue_id}/status",
    operation_id="get_queue_status",
    responses={
        200: {"model": SessionQueueAndProcessorStatus},
    },
)
async def get_queue_status(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionQueueAndProcessorStatus:
    """Gets the status of the session queue"""
    try:
        queue = ApiDependencies.invoker.services.session_queue.get_queue_status(queue_id)
        processor = ApiDependencies.invoker.services.session_processor.get_status()
        return SessionQueueAndProcessorStatus(queue=queue, processor=processor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting queue status: {e}")


@session_queue_router.get(
    "/{queue_id}/b/{batch_id}/status",
    operation_id="get_batch_status",
    responses={
        200: {"model": BatchStatus},
    },
)
async def get_batch_status(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch_id: str = Path(description="The batch to get the status of"),
) -> BatchStatus:
    """Gets the status of the session queue"""
    try:
        return ApiDependencies.invoker.services.session_queue.get_batch_status(queue_id=queue_id, batch_id=batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting batch status: {e}")


@session_queue_router.get(
    "/{queue_id}/i/{item_id}",
    operation_id="get_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
    response_model_exclude_none=True,
)
async def get_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: int = Path(description="The queue item to get"),
) -> SessionQueueItem:
    """Gets a queue item"""
    try:
        return ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)
    except SessionQueueItemNotFoundError:
        raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching queue item: {e}")


@session_queue_router.delete(
    "/{queue_id}/i/{item_id}",
    operation_id="delete_queue_item",
)
async def delete_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: int = Path(description="The queue item to delete"),
) -> None:
    """Deletes a queue item"""
    try:
        ApiDependencies.invoker.services.session_queue.delete_queue_item(item_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while deleting queue item: {e}")


@session_queue_router.put(
    "/{queue_id}/i/{item_id}/cancel",
    operation_id="cancel_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
)
async def cancel_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: int = Path(description="The queue item to cancel"),
) -> SessionQueueItem:
    """Deletes a queue item"""
    try:
        return ApiDependencies.invoker.services.session_queue.cancel_queue_item(item_id)
    except SessionQueueItemNotFoundError:
        raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling queue item: {e}")


@session_queue_router.get(
    "/{queue_id}/counts_by_destination",
    operation_id="counts_by_destination",
    responses={200: {"model": SessionQueueCountsByDestination}},
)
async def counts_by_destination(
    queue_id: str = Path(description="The queue id to query"),
    destination: str = Query(description="The destination to query"),
) -> SessionQueueCountsByDestination:
    """Gets the counts of queue items by destination"""
    try:
        return ApiDependencies.invoker.services.session_queue.get_counts_by_destination(
            queue_id=queue_id, destination=destination
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching counts by destination: {e}")


@session_queue_router.delete(
    "/{queue_id}/d/{destination}",
    operation_id="delete_by_destination",
    responses={200: {"model": DeleteByDestinationResult}},
)
async def delete_by_destination(
    queue_id: str = Path(description="The queue id to query"),
    destination: str = Path(description="The destination to query"),
) -> DeleteByDestinationResult:
    """Deletes all items with the given destination"""
    try:
        return ApiDependencies.invoker.services.session_queue.delete_by_destination(
            queue_id=queue_id, destination=destination
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while deleting by destination: {e}")
