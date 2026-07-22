"""Tests for the post-dequeue cancellation guard that closes the multi-GPU cancel-loss race.

A cancellation can mark a queue item terminal in the window between dequeue claiming it and the
worker recording `queue_item` (so the status-changed handler can't set the worker's cancel_event).
`_is_queue_item_terminal` is the fresh DB re-check the worker uses to skip running such an item.
"""

from types import SimpleNamespace

import pytest

from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItemNotFoundError


class _Queue:
    def __init__(self, status: str | None = None, raise_not_found: bool = False):
        self._status = status
        self._raise = raise_not_found

    def get_queue_item(self, item_id: int):
        if self._raise:
            raise SessionQueueItemNotFoundError("gone")
        return SimpleNamespace(item_id=item_id, status=self._status)


def _processor_with_queue(queue: _Queue) -> DefaultSessionProcessor:
    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(services=SimpleNamespace(session_queue=queue))  # type: ignore[attr-defined]
    return processor


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("in_progress", False),
        ("pending", False),
        ("canceled", True),
        ("failed", True),
        ("completed", True),
    ],
)
def test_is_queue_item_terminal_status(status: str, expected: bool):
    processor = _processor_with_queue(_Queue(status=status))
    assert processor._is_queue_item_terminal(1) is expected


def test_is_queue_item_terminal_treats_missing_as_terminal():
    # A deleted row (e.g. queue cleared during the race) should be treated as terminal, not run.
    processor = _processor_with_queue(_Queue(raise_not_found=True))
    assert processor._is_queue_item_terminal(1) is True


def test_post_dequeue_guard_cancels_stale_claimed_item():
    """A delayed cancellation handler for the PREVIOUS item can set the worker's cancel_event after
    it was cleared but before dequeue() replaced worker.queue_item. The guard then skips the freshly
    claimed row — which is 'in_progress' and had no cancellation of its own — so it must explicitly
    cancel it. Without that write the item is abandoned in_progress forever (dequeue only claims
    'pending' rows)."""
    from threading import Event
    from unittest.mock import MagicMock

    from invokeai.app.services.session_processor.session_processor_default import _SessionWorker

    stop_event = Event()
    resume_event = Event()
    resume_event.set()
    poll_now_event = Event()

    worker = _SessionWorker(device=None, runner=MagicMock())
    canceled: list[int] = []
    claimed = SimpleNamespace(item_id=42, session_id="sess", queue_id="default")

    class _RaceQueue:
        def __init__(self):
            self.claimed_once = False

        def dequeue(self, device=None):
            if self.claimed_once:
                stop_event.set()
                return None
            self.claimed_once = True
            # The delayed handler for the previous item fires here — after cancel_event.clear(),
            # before the guard runs.
            worker.cancel_event.set()
            return claimed

        def get_queue_item(self, item_id: int):
            return SimpleNamespace(item_id=item_id, status="canceled" if item_id in canceled else "in_progress")

        def cancel_queue_item(self, item_id: int):
            canceled.append(item_id)
            return SimpleNamespace(item_id=item_id, status="canceled")

    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(  # type: ignore[attr-defined]
        services=SimpleNamespace(session_queue=_RaceQueue(), logger=MagicMock(), image_moves=None)
    )
    processor._polling_interval = 0.001
    from threading import BoundedSemaphore

    processor._thread_semaphore = BoundedSemaphore(1)

    processor._process(worker=worker, stop_event=stop_event, poll_now_event=poll_now_event, resume_event=resume_event)

    assert canceled == [42]
