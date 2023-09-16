from typing import Optional

from fastapi import Body, Path, Query
from fastapi.routing import APIRouter

from invokeai.app.services.session_execution.session_execution_common import SessionExecutionStatusResult
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    CancelByBatchIDsResult,
    ClearResult,
    EnqueueBatchResult,
    EnqueueGraphResult,
    PruneResult,
    SessionQueueItem,
    SessionQueueItemDTO,
    SessionQueueStatusResult,
)
from invokeai.app.services.shared.models import CursorPaginatedResults

from ...services.graph import Graph
from ..dependencies import ApiDependencies

session_queue_router = APIRouter(prefix="/v1/queue", tags=["queue"])


class SessionQueueAndExecutionStatusResult(SessionQueueStatusResult, SessionExecutionStatusResult):
    pass


@session_queue_router.post(
    "/{queue_id}/enqueue_graph",
    operation_id="enqueue_graph",
    responses={
        201: {"model": EnqueueGraphResult},
    },
)
async def enqueue_graph(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    graph: Graph = Body(description="The graph to enqueue"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch in the queue"),
) -> EnqueueGraphResult:
    """Enqueues a graph for single execution."""

    return ApiDependencies.invoker.services.session_queue.enqueue_graph(queue_id=queue_id, graph=graph, prepend=prepend)


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
        queue_id=queue_id, limit=limit, status=status, order_id=cursor, priority=priority
    )


@session_queue_router.put(
    "/{queue_id}/start",
    operation_id="start",
)
async def start(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> None:
    """Starts session queue execution"""
    return ApiDependencies.invoker.services.session_execution.start(
        queue_id=queue_id,
    )


@session_queue_router.put(
    "/{queue_id}/stop",
    operation_id="stop",
)
async def stop(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> None:
    """Stops session queue execution, waiting for the currently executing session to finish"""
    return ApiDependencies.invoker.services.session_execution.stop(
        queue_id=queue_id,
    )


@session_queue_router.put(
    "/{queue_id}/cancel",
    operation_id="cancel",
)
async def cancel(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> None:
    """Stops session queue execution, immediately canceling the currently-executing session"""
    return ApiDependencies.invoker.services.session_execution.cancel(
        queue_id=queue_id,
    )


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
    current = ApiDependencies.invoker.services.session_execution.get_current(
        queue_id=queue_id,
    )
    if current is not None and current.batch_id in batch_ids:
        ApiDependencies.invoker.services.session_execution.cancel(
            queue_id=queue_id,
        )
    return ApiDependencies.invoker.services.session_queue.cancel_by_batch_ids(queue_id=queue_id, batch_ids=batch_ids)


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
    ApiDependencies.invoker.services.session_execution.cancel(
        queue_id=queue_id,
    )
    return ApiDependencies.invoker.services.session_queue.clear(
        queue_id=queue_id,
    )


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
    return ApiDependencies.invoker.services.session_queue.prune(
        queue_id=queue_id,
    )


@session_queue_router.get(
    "/{queue_id}/current",
    operation_id="current",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
async def current(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> Optional[SessionQueueItem]:
    """Gets the currently execution queue item"""
    return ApiDependencies.invoker.services.session_execution.get_current(
        queue_id=queue_id,
    )


@session_queue_router.get(
    "/{queue_id}/peek",
    operation_id="peek",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
async def peek(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> Optional[SessionQueueItem]:
    """Gets the next queue item, without executing it"""
    return ApiDependencies.invoker.services.session_queue.peek(
        queue_id=queue_id,
    )


@session_queue_router.get(
    "/{queue_id}/status",
    operation_id="get_status",
    responses={
        200: {"model": SessionQueueAndExecutionStatusResult},
    },
)
async def get_status(
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionQueueAndExecutionStatusResult:
    """Gets the status of the session queue"""
    queue_status = ApiDependencies.invoker.services.session_queue.get_status(
        queue_id=queue_id,
    )
    execution_status = ApiDependencies.invoker.services.session_execution.get_status(
        queue_id=queue_id,
    )

    return SessionQueueAndExecutionStatusResult(**queue_status.dict(), **execution_status.dict())


@session_queue_router.get(
    "/{queue_id}/i/{item_id}",
    operation_id="get_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
)
async def get_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: str = Path(description="The queue item to get"),
) -> SessionQueueItem:
    """Gets a queue item"""
    return ApiDependencies.invoker.services.session_queue.get_queue_item(queue_id=queue_id, item_id=item_id)


@session_queue_router.put(
    "/{queue_id}/i/{item_id}/cancel",
    operation_id="cancel_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
)
async def cancel_queue_item(
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: str = Path(description="The queue item to cancel"),
) -> SessionQueueItem:
    """Deletes a queue item"""
    queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(queue_id=queue_id, item_id=item_id)
    if queue_item.status not in ["completed", "failed", "canceled"]:
        return ApiDependencies.invoker.services.session_queue.set_queue_item_status(
            queue_id=queue_id, item_id=item_id, status="canceled"
        )
    return queue_item


@session_queue_router.put(
    "/{queue_id}/start_processor",
    operation_id="start_processor",
)
async def start_processor() -> None:
    """Deletes a queue item"""
    ApiDependencies.invoker.services.session_processor.start()


@session_queue_router.put(
    "/{queue_id}/stop_processor",
    operation_id="stop_processor",
)
async def stop_processor() -> None:
    """Deletes a queue item"""
    ApiDependencies.invoker.services.session_processor.stop()
