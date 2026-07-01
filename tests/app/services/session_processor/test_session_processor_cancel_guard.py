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
