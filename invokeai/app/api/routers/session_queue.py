from typing import Optional

from fastapi import Body, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    BatchStatus,
    CancelByBatchIDsResult,
    CancelByDestinationResult,
    ClearResult,
    EnqueueBatchResult,
    PruneResult,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemDTO,
    SessionQueueStatus,
)
from invokeai.app.services.shared.pagination import CursorPaginatedResults

session_queue_router = APIRouter(prefix="/v1/queue", tags=["queue"])


class SessionQueueAndProcessorStatus(BaseModel):
    """The overall status of session queue and processor"""

    queue: SessionQueueStatus
    processor: SessionProcessorStatus


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
) -> EnqueueBatchResult:
    """Processes a batch and enqueues the output graphs for execution."""

    return ApiDependencies.invoker.services.session_queue.enqueue_batch(queue_id=queue_id, batch=batch, prepend=prepend)


@session_queue_router.get(
    "/{queue_id}/list",
    operation_id="list_queue_items",
    responses={
        200: {"model": CursorPaginatedResults[SessionQueueItemDTO]},
    },
)
async def list_queue_items(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    limit: int = Query(default=50, description="The number of items to fetch"),
    status: Optional[QUEUE_ITEM_STATUS] = Query(default=None, description="The status of items to fetch"),
    cursor: Optional[int] = Query(default=None, description="The pagination cursor"),
    priority: int = Query(default=0, description="The pagination cursor priority"),
) -> CursorPaginatedResults[SessionQueueItemDTO]:
    """Gets all queue items (without graphs)"""

    return ApiDependencies.invoker.services.session_queue.list_queue_items(
        queue_id=queue_id, limit=limit, status=status, cursor=cursor, priority=priority
    )


@session_queue_router.put(
    "/{queue_id}/processor/resume",
    operation_id="resume",
    responses={200: {"model": SessionProcessorStatus}},
)
async def resume(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionProcessorStatus:
    """Resumes session processor"""
    return ApiDependencies.invoker.services.session_processor.resume()


@session_queue_router.put(
    "/{queue_id}/processor/pause",
    operation_id="pause",
    responses={200: {"model": SessionProcessorStatus}},
)
async def Pause(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionProcessorStatus:
    """Pauses session processor"""
    return ApiDependencies.invoker.services.session_processor.pause()


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
    return ApiDependencies.invoker.services.session_queue.cancel_by_batch_ids(queue_id=queue_id, batch_ids=batch_ids)


@session_queue_router.put(
    "/{queue_id}/cancel_by_destination",
    operation_id="cancel_by_destination",
    responses={200: {"model": CancelByBatchIDsResult}},
)
async def cancel_by_destination(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    destination: str = Query(description="The destination to cancel all queue items for"),
) -> CancelByDestinationResult:
    """Immediately cancels all queue items with the given origin"""
    return ApiDependencies.invoker.services.session_queue.cancel_by_destination(
        queue_id=queue_id, destination=destination
    )


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
    queue_item = ApiDependencies.invoker.services.session_queue.get_current(queue_id)
    if queue_item is not None:
        ApiDependencies.invoker.services.session_queue.cancel_queue_item(queue_item.item_id)
    clear_result = ApiDependencies.invoker.services.session_queue.clear(queue_id)
    return clear_result


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
    return ApiDependencies.invoker.services.session_queue.prune(queue_id)


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
    return ApiDependencies.invoker.services.session_queue.get_current(queue_id)


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
    return ApiDependencies.invoker.services.session_queue.get_next(queue_id)


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
    queue = ApiDependencies.invoker.services.session_queue.get_queue_status(queue_id)
    processor = ApiDependencies.invoker.services.session_processor.get_status()
    return SessionQueueAndProcessorStatus(queue=queue, processor=processor)


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
    return ApiDependencies.invoker.services.session_queue.get_batch_status(queue_id=queue_id, batch_id=batch_id)


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
    return ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)


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

    return ApiDependencies.invoker.services.session_queue.cancel_queue_item(item_id)


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
    return ApiDependencies.invoker.services.session_queue.get_counts_by_destination(
        queue_id=queue_id, destination=destination
    )
