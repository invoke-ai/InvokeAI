"""A worker whose GPU is lent to a borrower leaves queue items for a free GPU.

A session claimed onto a lent GPU would only wait for the borrow to end -- up to a whole prompt
rewrite for work outside the queue -- while another GPU may be idle. These drive the real worker loop
(`_process`) against the real device pool, with the queue and runner faked.
"""

import threading
from collections.abc import Iterator
from threading import BoundedSemaphore, Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    _SessionWorker,
)
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import TorchDevice

GPU0 = torch.device("cuda:0")
GPU1 = torch.device("cuda:1")


@pytest.fixture(autouse=True)
def pool() -> Iterator[None]:
    GENERATION_DEVICE_POOL.reset()
    GENERATION_DEVICE_POOL.set_generation_devices([GPU0, GPU1])
    # The worker pins torch to its GPU before its first run; there is no real second GPU here.
    with patch("invokeai.app.services.session_processor.session_processor_default.torch.cuda.set_device"):
        try:
            yield
        finally:
            TorchDevice.clear_session_device()
            GENERATION_DEVICE_POOL.reset()


def _start_worker(device: torch.device) -> tuple[threading.Thread, Event, list[str], DefaultSessionProcessor]:
    """Run one worker on ``device`` in a thread; it records each claim and stops after running one item."""
    stop_event = Event()
    resume_event = Event()
    resume_event.set()
    claims: list[str] = []
    item = SimpleNamespace(item_id=1, session_id="s", queue_id="default")

    class _Queue:
        def dequeue(self, device=None):
            claims.append(device)
            return item

        def get_queue_item(self, item_id: int):
            return SimpleNamespace(item_id=item_id, status="in_progress")

    runner = MagicMock()
    runner.workflow_call_queue_lifecycle.run_queue_item.side_effect = lambda queue_item: stop_event.set()
    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(  # type: ignore[attr-defined]
        services=SimpleNamespace(
            session_queue=_Queue(),
            logger=MagicMock(),
            image_moves=None,
            configuration=SimpleNamespace(multiuser=False, clear_vram_after_session=False),
        )
    )
    processor._polling_interval = 5  # long enough that only the release listener can wake it in time
    processor._thread_semaphore = BoundedSemaphore(1)
    processor._poll_now_event = Event()
    GENERATION_DEVICE_POOL.set_release_listener(processor._poll_now_event.set)
    thread = threading.Thread(
        target=processor._process,
        kwargs={
            "worker": _SessionWorker(device=device, runner=runner),
            "stop_event": stop_event,
            "poll_now_event": processor._poll_now_event,
            "resume_event": resume_event,
        },
        daemon=True,
    )
    thread.start()
    return thread, stop_event, claims, processor


def test_worker_leaves_items_alone_while_its_gpu_is_lent_and_resumes_on_release() -> None:
    lent = GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda")
    assert lent == GPU0

    thread, stop_event, claims, processor = _start_worker(GPU0)
    try:
        thread.join(timeout=0.3)
        assert claims == []  # nothing claimed onto the lent GPU

        GENERATION_DEVICE_POOL.release_borrow(GPU0)
        # The release wakes the worker at once; its 5 s poll interval would fail this join.
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert claims == ["cuda:0"]
    finally:
        stop_event.set()
        processor._poll_now_event.set()
        thread.join(timeout=2)


def test_worker_on_a_free_gpu_claims_while_another_is_lent() -> None:
    assert GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda") == GPU0

    thread, stop_event, claims, processor = _start_worker(GPU1)
    try:
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert claims == ["cuda:1"]
    finally:
        stop_event.set()
        processor._poll_now_event.set()
        thread.join(timeout=2)
        GENERATION_DEVICE_POOL.release_borrow(GPU0)
