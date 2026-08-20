import asyncio
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.invocations.fields import ImageField, VideoField
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    BatchStatus,
    CancelAllExceptCurrentResult,
    CancelByBatchIDsResult,
    CancelByDestinationResult,
    ClearResult,
    DeleteAllExceptCurrentResult,
    DeleteByDestinationResult,
    EnqueueBatchResult,
    ItemIdsResult,
    PruneResult,
    RetryItemsResult,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
    SessionQueueItemSummary,
    SessionQueueStatus,
)
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException

session_queue_router = APIRouter(prefix="/v1/queue", tags=["queue"])

# Upper bound on the number of item ids a client may ask about in one request. Without it a
# caller can post tens of thousands of ids, which the SQLite layer would either expand past the
# per-statement bind limit or grind through in a long-running query. The list is meant to cover
# the rows a client actually has on screen, so this is far above any legitimate use.
MAX_QUEUE_ITEM_IDS_PER_REQUEST = 1000


class SessionQueueAndProcessorStatus(BaseModel):
    """The overall status of session queue and processor"""

    queue: SessionQueueStatus
    processor: SessionProcessorStatus


def _image_record_exists(image_name: str) -> bool:
    return ApiDependencies.invoker.services.image_records.exists(image_name)


def _video_record_exists(video_name: str) -> bool:
    # video_records has no exists() the way image_records does, and widening that ABC would add
    # divergence from upstream for no gain here — get() already answers the question.
    try:
        ApiDependencies.invoker.services.video_records.get(video_name)
    except VideoRecordNotFoundException:
        return False
    return True


def strip_missing_image_results(
    queue_item: SessionQueueItem,
    image_exists: Callable[[str], bool] | None = None,
    video_exists: Callable[[str], bool] | None = None,
) -> SessionQueueItem:
    """Remove result outputs whose image or video records have been deleted.

    Completed queue history can outlive its output media. API clients hydrate
    images and videos listed in `session.results`; returning stale names makes them
    loop on 404s. Keep the queue item/history, but do not advertise impossible outputs.
    """
    if not queue_item.session.results:
        return queue_item

    image_exists = image_exists or _image_record_exists
    video_exists = video_exists or _video_record_exists
    filtered_results = {}
    did_filter = False
    image_cache: dict[str, bool] = {}
    video_cache: dict[str, bool] = {}

    def is_media_field(item: object) -> bool:
        return isinstance(item, (ImageField, VideoField))

    def cached_exists(field: ImageField | VideoField) -> bool:
        if isinstance(field, VideoField):
            if field.video_name not in video_cache:
                video_cache[field.video_name] = video_exists(field.video_name)
            return video_cache[field.video_name]
        if field.image_name not in image_cache:
            image_cache[field.image_name] = image_exists(field.image_name)
        return image_cache[field.image_name]

    for node_id, output in queue_item.session.results.items():
        image = getattr(output, "image", None)
        if isinstance(image, ImageField) and not cached_exists(image):
            did_filter = True
            continue

        video = getattr(output, "video", None)
        if isinstance(video, VideoField) and not cached_exists(video):
            did_filter = True
            continue

        collection = getattr(output, "collection", None)
        if isinstance(collection, list) and any(is_media_field(item) for item in collection):
            filtered_collection = [item for item in collection if not is_media_field(item) or cached_exists(item)]
            if len(filtered_collection) != len(collection):
                did_filter = True
                if len(filtered_collection) == 0:
                    continue
                output = output.model_copy(update={"collection": filtered_collection})

        filtered_results[node_id] = output

    if not did_filter:
        return queue_item

    sanitized_item = queue_item.model_copy(deep=True)
    sanitized_item.session.results = filtered_results
    return sanitized_item


def _get_workflow_call_root_queue_item(queue_item: SessionQueueItem) -> SessionQueueItem:
    if queue_item.root_item_id is None:
        return queue_item
    return ApiDependencies.invoker.services.session_queue.get_queue_item(queue_item.root_item_id)


# What a non-admin must not see on another user's queue item, and what each field is replaced
# with. One table for both the full item and the list summary: the two are different projections
# of the same row, and a second redaction list is how they drift apart - a field stripped from the
# list but left on the detail view is still leaked. Fields absent from a model are skipped, so the
# full-item-only entries below simply do not apply to the summary.
#
# Replacements are built per call so that no two sanitized items share one mutable object.
#
# `device` is deliberately not redacted: it names the GPU the instance ran the job on, which is a
# property of the hardware rather than of the other user's work, and the queue list has always
# shown it.
_REDACTIONS: dict[str, Callable[[], Any]] = {
    "user_id": lambda: "redacted",
    "user_display_name": lambda: None,
    "user_email": lambda: None,
    "batch_id": lambda: "redacted",
    "session_id": lambda: "redacted",
    "origin": lambda: None,
    "destination": lambda: None,
    "priority": lambda: 0,
    "field_values": lambda: None,
    "retried_from_item_id": lambda: None,
    "workflow_call_id": lambda: None,
    "parent_item_id": lambda: None,
    "parent_session_id": lambda: None,
    "root_item_id": lambda: None,
    "workflow_call_depth": lambda: None,
    "workflow": lambda: None,
    "error_type": lambda: None,
    "error_message": lambda: None,
    "error_traceback": lambda: None,
    "session": lambda: GraphExecutionState(id="redacted", graph=Graph()),
}

AnyQueueItem = TypeVar("AnyQueueItem", SessionQueueItem, SessionQueueItemSummary)


def sanitize_queue_item_for_user(queue_item: AnyQueueItem, current_user_id: str, is_admin: bool) -> AnyQueueItem:
    """Sanitize a queue item, or a queue item summary, for a non-admin viewing another user's item.

    Only item_id, queue_id, status, device and the timestamps survive; identity, generation
    parameters, graphs and workflows are stripped. Admins and the item's owner see everything.

    Args:
        queue_item: The queue item or summary to sanitize
        current_user_id: The ID of the current user viewing the item
        is_admin: Whether the current user is an admin

    Returns:
        The sanitized item (sensitive fields cleared if necessary)
    """
    # Admins and item owners can see everything
    if is_admin or queue_item.user_id == current_user_id:
        if isinstance(queue_item, SessionQueueItem):
            return strip_missing_image_results(queue_item)
        return queue_item

    updates = {
        field: build_replacement()
        for field, build_replacement in _REDACTIONS.items()
        if field in type(queue_item).model_fields
    }
    return queue_item.model_copy(update=updates)


def get_queue_item_for_mutation(queue_id: str, item_id: int, current_user: CurrentUserOrDefault) -> SessionQueueItem:
    queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)

    if queue_item.queue_id != queue_id:
        raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")

    if queue_item.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail=f"You do not have permission to mutate queue item {item_id}")

    return queue_item


@session_queue_router.post(
    "/{queue_id}/enqueue_batch",
    operation_id="enqueue_batch",
    responses={
        201: {"model": EnqueueBatchResult},
    },
)
async def enqueue_batch(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch: Batch = Body(description="Batch to process"),
    prepend: bool = Body(default=False, description="Whether or not to prepend this batch in the queue"),
) -> EnqueueBatchResult:
    """Processes a batch and enqueues the output graphs for execution for the current user."""
    await asyncio.to_thread(assert_image_move_maintenance_inactive)

    try:
        return await ApiDependencies.invoker.services.session_queue.enqueue_batch(
            queue_id=queue_id, batch=batch, prepend=prepend, user_id=current_user.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while enqueuing batch: {e}")


@session_queue_router.get(
    "/{queue_id}/list_all",
    operation_id="list_all_queue_items",
    responses={
        200: {"model": list[SessionQueueItem]},
    },
)
def list_all_queue_items(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    destination: Optional[str] = Query(default=None, description="The destination of queue items to fetch"),
) -> list[SessionQueueItem]:
    """Gets all queue items"""
    try:
        items = ApiDependencies.invoker.services.session_queue.list_all_queue_items(
            queue_id=queue_id,
            destination=destination,
        )
        # Sanitize items for non-admin users
        return [sanitize_queue_item_for_user(item, current_user.user_id, current_user.is_admin) for item in items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while listing all queue items: {e}")


@session_queue_router.get(
    "/{queue_id}/item_ids",
    operation_id="get_queue_item_ids",
    responses={
        200: {"model": ItemIdsResult},
    },
)
def get_queue_item_ids(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    origin_prefix: Optional[str] = Query(
        default=None, description="Only include queue items whose origin starts with this prefix"
    ),
) -> ItemIdsResult:
    """Gets all queue item ids that match the given parameters.

    IDs for every user's items are returned (item ids carry no sensitive data on their own).
    When the corresponding items are hydrated via get_queue_items_by_item_ids, those belonging
    to other users are redacted by sanitize_queue_item_for_user. This lets a non-admin see
    partially-redacted entries for other users' jobs in the queue list, while still revealing
    only timestamps and status for items they do not own.

    current_user is required so the endpoint stays behind authentication in multiuser mode.
    """
    try:
        return ApiDependencies.invoker.services.session_queue.get_queue_item_ids(
            queue_id=queue_id, order_dir=order_dir, origin_prefix=origin_prefix
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while listing all queue item ids: {e}")


@session_queue_router.post(
    "/{queue_id}/items_by_ids",
    operation_id="get_queue_items_by_item_ids",
    responses={200: {"model": list[SessionQueueItem]}},
)
def get_queue_items_by_item_ids(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_ids: list[int] = Body(
        embed=True,
        max_length=MAX_QUEUE_ITEM_IDS_PER_REQUEST,
        description="Object containing list of queue item ids to fetch queue items for",
    ),
) -> list[SessionQueueItem]:
    """Gets queue items for the specified queue item ids. Maintains order of item ids.

    Bound the legacy full-item endpoint as well as the summary endpoint: callers can otherwise
    force one graph deserialization and response copy per supplied id, defeating the queue-list
    optimization with an authenticated memory/CPU exhaustion request.
    """
    try:
        session_queue_service = ApiDependencies.invoker.services.session_queue

        # Fetch queue items preserving the order of requested item ids
        queue_items: list[SessionQueueItem] = []
        for item_id in item_ids:
            try:
                queue_item = session_queue_service.get_queue_item(item_id=item_id)
                if queue_item.queue_id != queue_id:  # Auth protection for items from other queues
                    continue
                # Sanitize item for non-admin users
                sanitized_item = sanitize_queue_item_for_user(queue_item, current_user.user_id, current_user.is_admin)
                queue_items.append(sanitized_item)
            except Exception:
                # Skip missing queue items - they may have been deleted between item id fetch and queue item fetch
                continue

        return queue_items
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get queue items")


@session_queue_router.post(
    "/{queue_id}/item_summaries_by_ids",
    operation_id="get_queue_item_summaries_by_ids",
    responses={200: {"model": list[SessionQueueItemSummary]}},
)
def get_queue_item_summaries_by_ids(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_ids: list[int] = Body(
        embed=True,
        max_length=MAX_QUEUE_ITEM_IDS_PER_REQUEST,
        description="Object containing list of queue item ids to fetch summaries for",
    ),
) -> list[SessionQueueItemSummary]:
    """Gets lightweight queue item summaries for specified IDs in requested order."""
    try:
        summaries = ApiDependencies.invoker.services.session_queue.get_queue_item_summaries_by_ids(
            queue_id=queue_id, item_ids=item_ids
        )
        return [sanitize_queue_item_for_user(item, current_user.user_id, current_user.is_admin) for item in summaries]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get queue item summaries")


@session_queue_router.put(
    "/{queue_id}/processor/resume",
    operation_id="resume",
    responses={200: {"model": SessionProcessorStatus}},
)
def resume(
    current_user: AdminUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionProcessorStatus:
    """Resumes session processor. Admin only."""
    try:
        return ApiDependencies.invoker.services.session_processor.resume()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while resuming queue: {e}")


@session_queue_router.put(
    "/{queue_id}/processor/pause",
    operation_id="pause",
    responses={200: {"model": SessionProcessorStatus}},
)
def pause(
    current_user: AdminUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> SessionProcessorStatus:
    """Pauses session processor. Admin only."""
    try:
        return ApiDependencies.invoker.services.session_processor.pause()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while pausing queue: {e}")


@session_queue_router.put(
    "/{queue_id}/cancel_all_except_current",
    operation_id="cancel_all_except_current",
    responses={200: {"model": CancelAllExceptCurrentResult}},
)
def cancel_all_except_current(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> CancelAllExceptCurrentResult:
    """Immediately cancels all queue items except in-processing items. Non-admin users can only cancel their own items."""
    try:
        # Admin users can cancel all items, non-admin users can only cancel their own
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.cancel_all_except_current(
            queue_id=queue_id, user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling all except current: {e}")


@session_queue_router.put(
    "/{queue_id}/delete_all_except_current",
    operation_id="delete_all_except_current",
    responses={200: {"model": DeleteAllExceptCurrentResult}},
)
def delete_all_except_current(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> DeleteAllExceptCurrentResult:
    """Immediately deletes all queue items except in-processing items. Non-admin users can only delete their own items."""
    try:
        # Admin users can delete all items, non-admin users can only delete their own
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.delete_all_except_current(
            queue_id=queue_id, user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while deleting all except current: {e}")


@session_queue_router.put(
    "/{queue_id}/cancel_by_batch_ids",
    operation_id="cancel_by_batch_ids",
    responses={200: {"model": CancelByBatchIDsResult}},
)
def cancel_by_batch_ids(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch_ids: list[str] = Body(description="The list of batch_ids to cancel all queue items for", embed=True),
) -> CancelByBatchIDsResult:
    """Immediately cancels all queue items from the given batch ids. Non-admin users can only cancel their own items."""
    try:
        # Admin users can cancel all items, non-admin users can only cancel their own
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.cancel_by_batch_ids(
            queue_id=queue_id, batch_ids=batch_ids, user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling by batch id: {e}")


@session_queue_router.put(
    "/{queue_id}/cancel_by_destination",
    operation_id="cancel_by_destination",
    responses={200: {"model": CancelByDestinationResult}},
)
def cancel_by_destination(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    destination: str = Query(description="The destination to cancel all queue items for"),
) -> CancelByDestinationResult:
    """Immediately cancels all queue items with the given destination. Non-admin users can only cancel their own items."""
    try:
        # Admin users can cancel all items, non-admin users can only cancel their own
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.cancel_by_destination(
            queue_id=queue_id, destination=destination, user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling by destination: {e}")


@session_queue_router.put(
    "/{queue_id}/retry_items_by_id",
    operation_id="retry_items_by_id",
    responses={200: {"model": RetryItemsResult}},
)
def retry_items_by_id(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_ids: list[int] = Body(description="The queue item ids to retry"),
) -> RetryItemsResult:
    """Retries the given queue items. Users can only retry their own items unless they are an admin."""
    try:
        # Check queue membership for all items and ownership for non-admins.
        valid_item_ids: list[int] = []
        for item_id in item_ids:
            try:
                queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)
                if queue_item.queue_id != queue_id:
                    raise HTTPException(
                        status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}"
                    )
                root_queue_item = _get_workflow_call_root_queue_item(queue_item)
                if root_queue_item.queue_id != queue_id:
                    raise HTTPException(
                        status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}"
                    )
                if not current_user.is_admin and root_queue_item.user_id != current_user.user_id:
                    raise HTTPException(
                        status_code=403, detail=f"You do not have permission to retry queue item {item_id}"
                    )
                valid_item_ids.append(item_id)
            except SessionQueueItemNotFoundError:
                # Skip items that don't exist - they will be handled by retry_items_by_id
                continue

        return ApiDependencies.invoker.services.session_queue.retry_items_by_id(
            queue_id=queue_id, item_ids=valid_item_ids
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while retrying queue items: {e}")


@session_queue_router.put(
    "/{queue_id}/clear",
    operation_id="clear",
    responses={
        200: {"model": ClearResult},
    },
)
def clear(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> ClearResult:
    """Clears the queue. Admin users clear (and cancel) all items; non-admin users clear only their
    own items — other users' queued and running items are untouched."""
    try:
        # The service cancels every in-progress item in scope itself (there can be several
        # with multiple workers), so there is no per-item authorization to do here: a
        # non-admin's scope is exactly their own items. The previous single get_current()
        # check both 403'd users whose arbitrary selected row belonged to someone else and
        # let a scoped clear interrupt another user's running generation.
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.clear(queue_id, user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while clearing queue: {e}")


@session_queue_router.put(
    "/{queue_id}/prune",
    operation_id="prune",
    responses={
        200: {"model": PruneResult},
    },
)
def prune(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
) -> PruneResult:
    """Prunes all completed or errored queue items. Non-admin users can only prune their own items."""
    try:
        # Admin users can prune all items, non-admin users can only prune their own
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.prune(queue_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while pruning queue: {e}")


@session_queue_router.get(
    "/{queue_id}/current",
    operation_id="get_current_queue_item",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
def get_current_queue_item(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    origin_prefix: Optional[str] = Query(
        default=None, description="Only include queue items whose origin starts with this prefix"
    ),
) -> Optional[SessionQueueItem]:
    """Gets the currently execution queue item"""
    try:
        item = ApiDependencies.invoker.services.session_queue.get_current(queue_id, origin_prefix=origin_prefix)
        if item is not None:
            item = sanitize_queue_item_for_user(item, current_user.user_id, current_user.is_admin)
        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting current queue item: {e}")


@session_queue_router.get(
    "/{queue_id}/next",
    operation_id="get_next_queue_item",
    responses={
        200: {"model": Optional[SessionQueueItem]},
    },
)
def get_next_queue_item(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    origin_prefix: Optional[str] = Query(
        default=None, description="Only include queue items whose origin starts with this prefix"
    ),
) -> Optional[SessionQueueItem]:
    """Gets the next queue item, without executing it"""
    try:
        item = ApiDependencies.invoker.services.session_queue.get_next(queue_id, origin_prefix=origin_prefix)
        if item is not None:
            item = sanitize_queue_item_for_user(item, current_user.user_id, current_user.is_admin)
        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting next queue item: {e}")


@session_queue_router.get(
    "/{queue_id}/status",
    operation_id="get_queue_status",
    responses={
        200: {"model": SessionQueueAndProcessorStatus},
    },
)
def get_queue_status(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    origin_prefix: Optional[str] = Query(
        default=None, description="Only include queue items whose origin starts with this prefix"
    ),
) -> SessionQueueAndProcessorStatus:
    """Gets the status of the session queue. Returns global counts; every user additionally gets
    their own pending/in_progress counts (so the UI can show an X/Y badge and scope personal UI
    like the progress bar to the user's own activity). Non-admin users cannot see the current
    item's identifiers unless they own it."""
    try:
        queue = ApiDependencies.invoker.services.session_queue.get_queue_status(
            queue_id,
            user_id=current_user.user_id,
            acting_user_id=current_user.user_id,
            origin_prefix=origin_prefix,
            is_admin=current_user.is_admin,
        )
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
def get_batch_status(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    batch_id: str = Path(description="The batch to get the status of"),
) -> BatchStatus:
    """Gets the status of a batch. Non-admin users only see their own batches."""
    try:
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.get_batch_status(
            queue_id=queue_id, batch_id=batch_id, user_id=user_id
        )
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
def get_queue_item(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: int = Path(description="The queue item to get"),
) -> SessionQueueItem:
    """Gets a queue item"""
    try:
        queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(item_id=item_id)
        if queue_item.queue_id != queue_id:
            raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")
        # Sanitize item for non-admin users
        return sanitize_queue_item_for_user(queue_item, current_user.user_id, current_user.is_admin)
    except SessionQueueItemNotFoundError:
        raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching queue item: {e}")


@session_queue_router.delete(
    "/{queue_id}/i/{item_id}",
    operation_id="delete_queue_item",
)
def delete_queue_item(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: int = Path(description="The queue item to delete"),
) -> None:
    """Deletes a queue item. Users can only delete their own items unless they are an admin."""
    try:
        # Get the queue item to check ownership
        queue_item = ApiDependencies.invoker.services.session_queue.get_queue_item(item_id)
        if queue_item.queue_id != queue_id:
            raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")

        root_queue_item = _get_workflow_call_root_queue_item(queue_item)
        if root_queue_item.queue_id != queue_id:
            raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")

        # The queue service deletes the entire chain, so authorization must use the root owner.
        if root_queue_item.user_id != current_user.user_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="You do not have permission to delete this queue item")

        ApiDependencies.invoker.services.session_queue.delete_queue_item(item_id)
    except SessionQueueItemNotFoundError:
        raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while deleting queue item: {e}")


@session_queue_router.put(
    "/{queue_id}/i/{item_id}/cancel",
    operation_id="cancel_queue_item",
    responses={
        200: {"model": SessionQueueItem},
    },
)
def cancel_queue_item(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to perform this operation on"),
    item_id: int = Path(description="The queue item to cancel"),
) -> SessionQueueItem:
    """Cancels a queue item. Users can only cancel their own items unless they are an admin."""
    try:
        get_queue_item_for_mutation(queue_id, item_id, current_user)

        return ApiDependencies.invoker.services.session_queue.cancel_queue_item(item_id)
    except SessionQueueItemNotFoundError:
        raise HTTPException(status_code=404, detail=f"Queue item with id {item_id} not found in queue {queue_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while canceling queue item: {e}")


@session_queue_router.get(
    "/{queue_id}/counts_by_destination",
    operation_id="counts_by_destination",
    responses={200: {"model": SessionQueueCountsByDestination}},
)
def counts_by_destination(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to query"),
    destination: str = Query(description="The destination to query"),
) -> SessionQueueCountsByDestination:
    """Gets the counts of queue items by destination. Non-admin users only see their own items."""
    try:
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.get_counts_by_destination(
            queue_id=queue_id, destination=destination, user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching counts by destination: {e}")


@session_queue_router.delete(
    "/{queue_id}/d/{destination}",
    operation_id="delete_by_destination",
    responses={200: {"model": DeleteByDestinationResult}},
)
def delete_by_destination(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(description="The queue id to query"),
    destination: str = Path(description="The destination to query"),
) -> DeleteByDestinationResult:
    """Deletes all items with the given destination. Non-admin users can only delete their own items."""
    try:
        # Admin users can delete all items, non-admin users can only delete their own
        user_id = None if current_user.is_admin else current_user.user_id
        return ApiDependencies.invoker.services.session_queue.delete_by_destination(
            queue_id=queue_id, destination=destination, user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while deleting by destination: {e}")
