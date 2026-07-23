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


def _run_guard_scenario(statuses: list[str], set_cancel_event: bool, set_stop_on_claim: bool = False):
    """Drives one _process claim through the post-dequeue guard.

    `statuses` is the sequence of statuses returned by successive get_queue_item calls for the
    claimed row (the last value repeats). `set_cancel_event` simulates the delayed handler for the
    PREVIOUS item firing between cancel_event.clear() and the guard. Returns
    (canceled_ids, run_items, worker).
    """
    from threading import BoundedSemaphore, Event
    from unittest.mock import MagicMock

    from invokeai.app.services.session_processor.session_processor_default import _SessionWorker

    stop_event = Event()
    resume_event = Event()
    resume_event.set()
    poll_now_event = Event()

    run_items: list[int] = []
    runner = MagicMock()
    runner.workflow_call_queue_lifecycle.run_queue_item.side_effect = lambda item: run_items.append(item.item_id)
    worker = _SessionWorker(device=None, runner=runner)
    canceled: list[int] = []
    claimed = SimpleNamespace(item_id=42, session_id="sess", queue_id="default")

    class _RaceQueue:
        def __init__(self):
            self.claimed_once = False
            self.status_reads = 0

        def dequeue(self, device=None):
            if self.claimed_once:
                stop_event.set()
                return None
            self.claimed_once = True
            # The delayed handler for the previous item fires here — after cancel_event.clear(),
            # before the guard runs.
            if set_cancel_event:
                worker.cancel_event.set()
            if set_stop_on_claim:
                stop_event.set()
            return claimed

        def get_queue_item(self, item_id: int):
            status = statuses[min(self.status_reads, len(statuses) - 1)]
            self.status_reads += 1
            return SimpleNamespace(item_id=item_id, status=status)

        def cancel_queue_item(self, item_id: int):
            canceled.append(item_id)
            return SimpleNamespace(item_id=item_id, status="canceled")

    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(  # type: ignore[attr-defined]
        services=SimpleNamespace(session_queue=_RaceQueue(), logger=MagicMock(), image_moves=None)
    )
    processor._polling_interval = 0.001
    processor._thread_semaphore = BoundedSemaphore(1)

    processor._process(worker=worker, stop_event=stop_event, poll_now_event=poll_now_event, resume_event=resume_event)
    return canceled, run_items, worker


def test_stale_cancel_event_does_not_discard_fresh_claim():
    """A delayed cancellation handler for the PREVIOUS item can set the worker's cancel_event after
    it was cleared but before dequeue() replaced worker.queue_item. No cancellation targeted the
    freshly claimed row (its status is 'in_progress' — a genuine cancel writes the row terminal
    before emitting), so the guard must clear the stale signal and RUN the item. Previously it
    canceled the unrelated item, silently discarding another user's queued generation (JPPhoto
    merge blocker, 2026-07-22)."""
    canceled, run_items, worker = _run_guard_scenario(statuses=["in_progress"], set_cancel_event=True)

    assert run_items == [42]
    assert canceled == []
    # The stale signal was cleared before the run and nothing re-set it.
    assert not worker.cancel_event.is_set()


def test_genuine_cancel_race_skips_terminal_item():
    """A cancel that landed between dequeue's claim and the worker recording queue_item has already
    marked the row terminal — the item must be skipped, not run, and needs no further cancel."""
    canceled, run_items, _worker = _run_guard_scenario(statuses=["canceled"], set_cancel_event=True)

    assert run_items == []
    assert canceled == []


def test_cancel_landing_between_terminal_check_and_clear_is_not_lost():
    """If a genuine cancel for the claimed item lands after the first terminal check but before the
    stale-event clear, the re-check after the clear must catch it (the DB write precedes the event),
    so the item is skipped rather than run with its cancellation wiped."""
    canceled, run_items, _worker = _run_guard_scenario(statuses=["in_progress", "canceled"], set_cancel_event=True)

    assert run_items == []
    assert canceled == []


def test_queue_cleared_event_only_cancels_workers_of_the_cleared_user():
    """A user-scoped clear must not stop other users' running sessions: the QueueClearedEvent
    carries the scoping user_id, and only workers running that user's items may be canceled.
    Previously every worker on the queue was canceled, stopping Bob's session while his row
    (out of the clear's scope) stayed in_progress — abandoned (JPPhoto merge blocker,
    2026-07-22)."""
    import asyncio
    from threading import Event
    from unittest.mock import MagicMock

    from invokeai.app.services.events.events_common import QueueClearedEvent
    from invokeai.app.services.session_processor.session_processor_default import _SessionWorker

    processor = DefaultSessionProcessor()
    processor._poll_now_event = Event()
    alice_worker = _SessionWorker(device=None, runner=MagicMock())
    alice_worker.queue_item = SimpleNamespace(item_id=1, queue_id="default", user_id="alice")
    bob_worker = _SessionWorker(device=None, runner=MagicMock())
    bob_worker.queue_item = SimpleNamespace(item_id=2, queue_id="default", user_id="bob")
    idle_worker = _SessionWorker(device=None, runner=MagicMock())
    idle_worker.queue_item = None
    processor._workers = [alice_worker, bob_worker, idle_worker]

    asyncio.run(processor._on_queue_cleared(("queue_cleared", QueueClearedEvent.build("default", user_id="alice"))))

    assert alice_worker.cancel_event.is_set()
    assert not bob_worker.cancel_event.is_set()
    assert not idle_worker.cancel_event.is_set()

    # An unscoped (admin) clear cancels every worker on the queue.
    alice_worker.cancel_event.clear()
    asyncio.run(processor._on_queue_cleared(("queue_cleared", QueueClearedEvent.build("default", user_id=None))))
    assert alice_worker.cancel_event.is_set()
    assert bob_worker.cancel_event.is_set()


def test_shutdown_race_cancels_fresh_claim_instead_of_running_it():
    """When the processor is stopping, a claim that raced the shutdown must not start new work —
    and must be marked canceled so it isn't abandoned in_progress (dequeue only claims 'pending')."""
    canceled, run_items, _worker = _run_guard_scenario(
        statuses=["in_progress"], set_cancel_event=True, set_stop_on_claim=True
    )

    assert run_items == []
    assert canceled == [42]
