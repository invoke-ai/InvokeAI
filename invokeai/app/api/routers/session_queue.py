from typing import Optional

from fastapi import Body, Path, Query
from fastapi.routing import APIRouter

from invokeai.app.services.session_execution.session_execution_common import SessionExecutionStatusResult
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    ClearResult,
    EnqueueResult,
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
    "/enqueue",
    operation_id="enqueue",
    responses={
        201: {"model": EnqueueResult},
    },
)
async def enqueue(
    graph: Graph = Body(description="The graph to enqueue"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch in the queue"),
) -> EnqueueResult:
    """Enqueues a graph for single execution."""

    return ApiDependencies.invoker.services.session_queue.enqueue(graph=graph, prepend=prepend)


@session_queue_router.post(
    "/enqueue_batch",
    operation_id="enqueue_batch",
    responses={
        201: {"model": EnqueueResult},
    },
)
async def enqueue_batch(
    batch: Batch = Body(description="Batch to process"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch in the queue"),
) -> EnqueueResult:
    """Processes a batch and enqueues the output graphs for execution."""

    return ApiDependencies.invoker.services.session_queue.enqueue_batch(batch=batch, prepend=prepend)


@session_queue_router.get(
    "/list",
    operation_id="list_queue_items",
    responses={
        200: {"model": CursorPaginatedResults[SessionQueueItemDTO]},
    },
)
async def list_queue_items(
    limit: int = Query(default=50, description="The number of items to fetch"),
    status: Optional[QUEUE_ITEM_STATUS] = Query(default=None, description="The status of items to fetch"),
    cursor: Optional[int] = Query(default=None, description="The pagination cursor"),
    priority: int = Query(default=0, description="The pagination cursor priority"),
) -> CursorPaginatedResults[SessionQueueItemDTO]:
    """Gets all queue items (without graphs)"""

    return ApiDependencies.invoker.services.session_queue.list_queue_items(
        limit=limit, status=status, cursor=cursor, priority=priority
    )


@session_queue_router.put(
    "/start",
    operation_id="start",
)
async def start() -> None:
    """Starts session queue execution"""
    return ApiDependencies.invoker.services.session_execution.start()


@session_queue_router.put(
    "/stop",
    operation_id="stop",
)
async def stop() -> None:
    """Stops session queue execution, waiting for the currently executing session to finish"""
    return ApiDependencies.invoker.services.session_execution.stop()


@session_queue_router.put(
    "/cancel",
    operation_id="cancel",
)
async def cancel() -> None:
    """Stops session queue execution, immediately canceling the currently-executing session"""
    return ApiDependencies.invoker.services.session_execution.cancel()


@session_queue_router.put(
    "/clear",
    operation_id="clear",
    responses={
        200: {"model": ClearResult},
    },
)
async def clear() -> ClearResult:
    """Clears the queue entirely, immediately canceling the currently-executing session"""
    ApiDependencies.invoker.services.session_execution.cancel()
    return ApiDependencies.invoker.services.session_queue.clear()


@session_queue_router.put(
    "/prune",
    operation_id="prune",
    responses={
        200: {"model": PruneResult},
    },
)
async def prune() -> PruneResult:
    """Prunes all completed or errored queue items"""
    return ApiDependencies.invoker.services.session_queue.prune()


@session_queue_router.get(
    "/current",
    operation_id="current",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
async def current() -> Optional[SessionQueueItem]:
    """Gets the currently execution queue item"""
    return ApiDependencies.invoker.services.session_execution.get_current()


@session_queue_router.get(
    "/peek",
    operation_id="peek",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
async def peek() -> Optional[SessionQueueItem]:
    """Gets the next queue item, without executing it"""
    return ApiDependencies.invoker.services.session_queue.peek()


@session_queue_router.get(
    "/status",
    operation_id="get_status",
    responses={
        200: {"model": SessionQueueAndExecutionStatusResult},
    },
)
async def get_status() -> SessionQueueAndExecutionStatusResult:
    """Gets the status of the session queue"""
    queue_status = ApiDependencies.invoker.services.session_queue.get_status()
    execution_status = ApiDependencies.invoker.services.session_execution.get_status()

    return SessionQueueAndExecutionStatusResult(**queue_status.dict(), **execution_status.dict())


@session_queue_router.get(
    "/q/{id}",
    operation_id="get_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
)
async def get_queue_item(id: int = Path(description="The queue item to get")) -> SessionQueueItem:
    """Gets a queue item"""
    return ApiDependencies.invoker.services.session_queue.get_queue_item(id=id)


@session_queue_router.put(
    "/q/{id}/cancel",
    operation_id="cancel_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
)
async def cancel_queue_item(
    id: int = Path(description="The queue item to cancel"),
) -> SessionQueueItem:
    """Deletes a queue item"""
    return ApiDependencies.invoker.services.session_queue.set_queue_item_status(id, "canceled")
