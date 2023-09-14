from abc import ABC, abstractmethod
from typing import Optional
from invokeai.app.services.graph import Graph

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    ClearResult,
    EnqueueResult,
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
    def enqueue(self, graph: Graph, prepend: bool) -> EnqueueResult:
        """Enqueues a single graph for execution."""
        pass

    @abstractmethod
    def enqueue_batch(self, batch: Batch, prepend: bool) -> EnqueueResult:
        """Enqueues all permutations of a batch for execution."""
        pass

    @abstractmethod
    def dequeue(self) -> Optional[SessionQueueItem]:
        """Dequeues the next session queue item, returning it if one is available."""
        pass

    @abstractmethod
    def peek(self) -> Optional[SessionQueueItem]:
        """Peeks at the next session queue item, returning it if one is available."""
        pass

    @abstractmethod
    def clear(self) -> ClearResult:
        """Deletes all session queue items"""
        pass

    @abstractmethod
    def prune(self) -> PruneResult:
        """Deletes all completed and errored session queue items"""
        pass

    @abstractmethod
    def is_empty(self) -> IsEmptyResult:
        """Checks if the queue is empty"""
        pass

    @abstractmethod
    def is_full(self) -> IsFullResult:
        """Checks if the queue is empty"""
        pass

    @abstractmethod
    def get_status(self) -> SessionQueueStatusResult:
        """Gets the number of queue items with each status"""
        pass

    @abstractmethod
    def list_queue_items(
        self,
        limit: int,
        priority: int,
        cursor: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
    ) -> CursorPaginatedResults[SessionQueueItemDTO]:
        """Gets a page of session queue items"""
        pass

    @abstractmethod
    def get_queue_item(self, id: int) -> SessionQueueItem:
        """Gets a session queue item by ID"""
        pass

    @abstractmethod
    def get_queue_item_by_session_id(self, session_id: str) -> SessionQueueItem:
        """Gets a queue item by session ID"""
        pass

    @abstractmethod
    def set_queue_item_status(self, id: int, status: QUEUE_ITEM_STATUS) -> SessionQueueItem:
        """Sets the status of a session queue item"""
        pass

    @abstractmethod
    def set_many_queue_item_status(self, ids: list[str], status: QUEUE_ITEM_STATUS) -> SetManyQueueItemStatusResult:
        """Sets the status of a session queue item"""
        pass

    @abstractmethod
    def delete_queue_item(self, id: int) -> SessionQueueItem:
        """Deletes a session queue item by ID"""
        pass
