from abc import ABC, abstractmethod
from typing import Any, Coroutine, Optional

from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    Batch,
    BatchStatus,
    CancelAllExceptCurrentResult,
    CancelByBatchIDsResult,
    CancelByDestinationResult,
    CancelByQueueIDResult,
    ClearResult,
    EnqueueBatchResult,
    IsEmptyResult,
    IsFullResult,
    PruneResult,
    RetryItemsResult,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemDTO,
    SessionQueueStatus,
)
from invokeai.app.services.shared.graph import GraphExecutionState
from invokeai.app.services.shared.pagination import CursorPaginatedResults


class SessionQueueBase(ABC):
    """Base class for session queue"""

    @abstractmethod
    def dequeue(self) -> Optional[SessionQueueItem]:
        """Dequeues the next session queue item."""
        pass

    @abstractmethod
    def enqueue_batch(self, queue_id: str, batch: Batch, prepend: bool) -> Coroutine[Any, Any, EnqueueBatchResult]:
        """Enqueues all permutations of a batch for execution."""
        pass

    @abstractmethod
    def get_current(self, queue_id: str) -> Optional[SessionQueueItem]:
        """Gets the currently-executing session queue item"""
        pass

    @abstractmethod
    def get_next(self, queue_id: str) -> Optional[SessionQueueItem]:
        """Gets the next session queue item (does not dequeue it)"""
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
    def get_queue_status(self, queue_id: str) -> SessionQueueStatus:
        """Gets the status of the queue"""
        pass

    @abstractmethod
    def get_counts_by_destination(self, queue_id: str, destination: str) -> SessionQueueCountsByDestination:
        """Gets the counts of queue items by destination"""
        pass

    @abstractmethod
    def get_batch_status(self, queue_id: str, batch_id: str) -> BatchStatus:
        """Gets the status of a batch"""
        pass

    @abstractmethod
    def complete_queue_item(self, item_id: int) -> SessionQueueItem:
        """Completes a session queue item"""
        pass

    @abstractmethod
    def cancel_queue_item(self, item_id: int) -> SessionQueueItem:
        """Cancels a session queue item"""
        pass

    @abstractmethod
    def fail_queue_item(
        self, item_id: int, error_type: str, error_message: str, error_traceback: str
    ) -> SessionQueueItem:
        """Fails a session queue item"""
        pass

    @abstractmethod
    def cancel_by_batch_ids(self, queue_id: str, batch_ids: list[str]) -> CancelByBatchIDsResult:
        """Cancels all queue items with matching batch IDs"""
        pass

    @abstractmethod
    def cancel_by_destination(self, queue_id: str, destination: str) -> CancelByDestinationResult:
        """Cancels all queue items with the given batch destination"""
        pass

    @abstractmethod
    def cancel_by_queue_id(self, queue_id: str) -> CancelByQueueIDResult:
        """Cancels all queue items with matching queue ID"""
        pass

    @abstractmethod
    def cancel_all_except_current(self, queue_id: str) -> CancelAllExceptCurrentResult:
        """Cancels all queue items except in-progress items"""
        pass

    @abstractmethod
    def list_queue_items(
        self,
        queue_id: str,
        limit: int,
        priority: int,
        cursor: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
    ) -> CursorPaginatedResults[SessionQueueItemDTO]:
        """Gets a page of session queue items"""
        pass

    @abstractmethod
    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        """Gets a session queue item by ID"""
        pass

    @abstractmethod
    def set_queue_item_session(self, item_id: int, session: GraphExecutionState) -> SessionQueueItem:
        """Sets the session for a session queue item. Use this to update the session state."""
        pass

    @abstractmethod
    def retry_items_by_id(self, queue_id: str, item_ids: list[int]) -> RetryItemsResult:
        """Retries the given queue items"""
        pass
