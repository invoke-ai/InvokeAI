import time
from dataclasses import dataclass
from threading import Lock

from invokeai.app.services.events.events_common import InvocationProgressEvent
from invokeai.app.services.progress_previews.progress_previews_base import ProgressPreviewsBase

# The floor between two image-bearing frames of one queue item. Ten frames a second is well past
# what a viewer can follow; above it every step became a JPEG encode on the generation thread and a
# socket emit for nothing. A final frame (percentage 1.0) and indeterminate frames always pass.
PROGRESS_IMAGE_MIN_INTERVAL_S = 0.1

# Revision counters outlive their frames (see `clear`). They are tiny, but bounded all the same:
# past this many frameless entries the oldest are forgotten.
COUNTER_RETENTION_LIMIT = 4096


@dataclass
class _ItemPreview:
    session_id: str
    revision: int = 0
    last_emitted_at: float | None = None
    event: InvocationProgressEvent | None = None
    invocation_id: str | None = None


class MemoryProgressPreviews(ProgressPreviewsBase):
    """In-memory previews: one entry per queue item, shared by every worker thread."""

    def __init__(self, min_interval_s: float = PROGRESS_IMAGE_MIN_INTERVAL_S, clock=time.monotonic) -> None:
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._lock = Lock()
        self._items: dict[int, _ItemPreview] = {}

    def reserve_revision(self, item_id: int, session_id: str, percentage: float | None) -> int | None:
        now = self._clock()
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.session_id != session_id:
                item = _ItemPreview(session_id=session_id)
                self._items[item_id] = item
            is_final = percentage is None or percentage >= 1.0
            if not is_final and item.last_emitted_at is not None and now - item.last_emitted_at < self._min_interval_s:
                return None
            item.revision += 1
            item.last_emitted_at = now
            return item.revision

    def record(self, event: InvocationProgressEvent) -> None:
        with self._lock:
            item = self._items.get(event.item_id)
            if item is None or item.session_id != event.session_id:
                # Recorded without a reservation (or after a session change): adopt it as-is.
                item = _ItemPreview(session_id=event.session_id, revision=event.revision or 0)
                self._items[event.item_id] = item
            item.event = event
            item.invocation_id = event.invocation.id

    def get(self, item_id: int) -> InvocationProgressEvent | None:
        with self._lock:
            item = self._items.get(item_id)
            return item.event if item is not None else None

    def list_for_user(self, user_id: str, queue_id: str | None = None) -> list[InvocationProgressEvent]:
        with self._lock:
            events = [item.event for item in self._items.values() if item.event is not None]
        return [
            event
            for event in sorted(events, key=lambda event: event.item_id)
            if event.user_id == user_id and (queue_id is None or event.queue_id == queue_id)
        ]

    def clear_node(self, item_id: int, invocation_id: str) -> None:
        with self._lock:
            item = self._items.get(item_id)
            if item is not None and item.invocation_id == invocation_id:
                item.event = None
                item.invocation_id = None

    def clear(self, item_id: int) -> None:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return
            # The frame goes; the counter stays so a later leg of the same item and session (a
            # workflow-call parent resuming after its child) keeps issuing higher revisions than
            # any a client already holds. The throttle does not carry over.
            item.event = None
            item.invocation_id = None
            item.last_emitted_at = None
            self._evict_stale_counters()

    def clear_all(self) -> None:
        with self._lock:
            self._items.clear()

    def _evict_stale_counters(self) -> None:
        """Caller holds the lock. Drop the oldest frameless entries past the retention cap."""
        excess = len(self._items) - COUNTER_RETENTION_LIMIT
        if excess <= 0:
            return
        for item_id in [item_id for item_id, item in self._items.items() if item.event is None][:excess]:
            del self._items[item_id]
