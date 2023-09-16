from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.graph import Graph
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    CancelByBatchIDsResult,
    ClearResult,
    EnqueueBatchResult,
    EnqueueGraphResult,
    IsEmptyResult,
    IsFullResult,
    PruneResult,
    SessionQueueItem,
    SessionQueueItemDTO,
    SessionQueueStatusResult,
    SetManyQueueItemStatusResult,
)
from invokeai.app.services.shared.models import CursorPaginatedResults


class SessionQueueBase(ABC):
    @abstractmethod
    def start_service(self, invoker: Invoker) -> None:
        """Startup callback for the SessionQueue service"""
        pass

    @abstractmethod
    def enqueue_graph(self, queue_id: str, graph: Graph, prepend: bool) -> EnqueueGraphResult:
        """Enqueues a single graph for execution."""
        pass

    @abstractmethod
    def enqueue_batch(self, queue_id: str, batch: Batch, prepend: bool) -> EnqueueBatchResult:
        """Enqueues all permutations of a batch for execution."""
        pass

    @abstractmethod
    def dequeue(self, queue_id: str) -> Optional[SessionQueueItem]:
        """Dequeues the next session queue item, returning it if one is available."""
        pass

    @abstractmethod
    def peek(self, queue_id: str) -> Optional[SessionQueueItem]:
        """Peeks at the next session queue item, returning it if one is available."""
        pass

    @abstractmethod
    def clear(self, queue_id: str) -> ClearResult:
        """Deletes all session queue items"""
        pass

    @abstractmethod
    def prune(self, queue_id: str) -> PruneResult:
        """Deletes all completed and errored session queue items"""
        pass

    @abstractmethod
    def is_empty(self, queue_id: str) -> IsEmptyResult:
        """Checks if the queue is empty"""
        pass

    @abstractmethod
    def is_full(self, queue_id: str) -> IsFullResult:
        """Checks if the queue is empty"""
        pass

    @abstractmethod
    def cancel_by_batch_ids(self, queue_id: str, batch_ids: list[str]) -> CancelByBatchIDsResult:
        """Cancels all queue items with matching batch IDs"""
        pass

    @abstractmethod
    def get_status(self, queue_id: str) -> SessionQueueStatusResult:
        """Gets the number of queue items with each status"""
        pass

    @abstractmethod
    def list_queue_items(
        self,
        queue_id: str,
        limit: int,
        priority: int,
        order_id: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
    ) -> CursorPaginatedResults[SessionQueueItemDTO]:
        """Gets a page of session queue items"""
        pass

    @abstractmethod
    def get_queue_item(self, queue_id: str, item_id: str) -> SessionQueueItem:
        """Gets a session queue item by ID"""
        pass

    @abstractmethod
    def get_queue_item_by_session_id(self, session_id: str) -> SessionQueueItem:
        """Gets a queue item by session ID"""
        pass

    @abstractmethod
    def set_queue_item_status(self, queue_id: str, item_id: str, status: QUEUE_ITEM_STATUS) -> SessionQueueItem:
        """Sets the status of a session queue item"""
        pass

    @abstractmethod
    def set_many_queue_item_status(
        self, queue_id: str, item_ids: list[str], status: QUEUE_ITEM_STATUS
    ) -> SetManyQueueItemStatusResult:
        """Sets the status of a session queue item"""
        pass

    @abstractmethod
    def delete_queue_item(self, queue_id: str, item_id: str) -> SessionQueueItem:
        """Deletes a session queue item by ID"""
        pass
