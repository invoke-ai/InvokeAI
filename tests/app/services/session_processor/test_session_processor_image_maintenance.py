"""Queue claims serialize with image storage maintenance reservations."""

import threading
from pathlib import Path
from types import SimpleNamespace

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import ImageMoveQueueActive, ImageMoveService
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor, _SessionWorker
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger


class _Queue:
    def __init__(self, pending: int) -> None:
        self.status = SimpleNamespace(pending=pending, in_progress=0)
        self.dequeue_started = threading.Event()
        self.allow_dequeue = threading.Event()
        self.dequeued = 0
        self.item = SimpleNamespace(item_id=1, session_id="session-1", queue_id="default")

    def has_active_queue_work(self) -> bool:
        return self.status.pending > 0 or self.status.in_progress > 0

    def dequeue(self, device: str | None = None):
        self.dequeued += 1
        self.dequeue_started.set()
        if not self.allow_dequeue.wait(timeout=10):
            raise AssertionError("queue claim was not released")
        self.status = SimpleNamespace(pending=0, in_progress=1)
        return self.item


def _services(tmp_path: Path, queue: _Queue) -> tuple[DefaultSessionProcessor, ImageMoveService]:
    config = InvokeAIAppConfig(use_memory_db=True)
    config._root = tmp_path
    logger = InvokeAILogger.get_logger(config=config)
    image_files = DiskImageFileStorage(tmp_path / "images")
    db = init_db(config=config, logger=logger, image_files=image_files)
    moves = ImageMoveService(db=db, image_files=image_files, config=config, logger=logger)
    invoker = SimpleNamespace(services=SimpleNamespace(session_queue=queue))
    moves.start(invoker)

    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(  # type: ignore[attr-defined]
        services=SimpleNamespace(session_queue=queue, image_moves=moves)
    )
    return processor, moves


def test_in_flight_queue_claim_is_visible_before_gallery_maintenance_can_start(tmp_path: Path) -> None:
    queue = _Queue(pending=1)
    queue.allow_dequeue.clear()
    processor, moves = _services(tmp_path, queue)
    claimed: list[tuple[bool, object | None]] = []
    claim_errors: list[Exception] = []

    def claim() -> None:
        try:
            claimed.append(processor._dequeue_if_storage_maintenance_inactive(device=None))
        except Exception as error:
            claim_errors.append(error)

    claim_thread = threading.Thread(target=claim)
    claim_thread.start()
    reservation_errors: list[Exception] = []
    reservation_thread: threading.Thread | None = None

    def reserve() -> None:
        try:
            with moves.reserve_gallery_maintenance():
                reservation_errors.append(AssertionError("maintenance ran while a queue item was in progress"))
        except Exception as error:
            reservation_errors.append(error)

    try:
        assert queue.dequeue_started.wait(timeout=10), "worker did not reach dequeue"

        reservation_attempted = threading.Event()
        real_mutation_lock = moves.image_mutation_lock

        def observe_reservation_lock():
            reservation_attempted.set()
            return real_mutation_lock()

        moves.image_mutation_lock = observe_reservation_lock  # type: ignore[method-assign]
        reservation_thread = threading.Thread(target=reserve)
        reservation_thread.start()
        assert reservation_attempted.wait(timeout=10), "maintenance did not attempt the shared mutation lock"
        assert moves.is_maintenance_active() is True
    finally:
        queue.allow_dequeue.set()
        claim_thread.join(timeout=10)
        if reservation_thread is not None:
            reservation_thread.join(timeout=10)
        moves.stop()

    assert not claim_thread.is_alive()
    assert reservation_thread is not None
    assert not reservation_thread.is_alive()
    assert claim_errors == []
    assert claimed == [(True, queue.item)]
    assert len(reservation_errors) == 1
    assert isinstance(reservation_errors[0], ImageMoveQueueActive)
    assert queue.status.in_progress == 1
    assert moves.is_maintenance_active() is False


def test_queue_work_admitted_after_reservation_waits_without_being_claimed(tmp_path: Path) -> None:
    queue = _Queue(pending=0)
    processor, moves = _services(tmp_path, queue)

    try:
        with moves.reserve_gallery_maintenance():
            # The API may admit an enqueue that raced the operation reservation; workers must
            # still leave it pending until the guarded filesystem scan is complete.
            queue.status = SimpleNamespace(pending=1, in_progress=0)
            allowed, item = processor._dequeue_if_storage_maintenance_inactive(device=None)

            assert allowed is False
            assert item is None
            assert queue.dequeued == 0
            assert queue.status.pending == 1
    finally:
        moves.stop()


def test_worker_stops_while_gallery_maintenance_holds_the_mutation_lock(tmp_path: Path) -> None:
    queue = _Queue(pending=0)
    queue.allow_dequeue.set()
    processor, moves = _services(tmp_path, queue)
    maintenance_entered = threading.Event()
    release_maintenance = threading.Event()
    maintenance_errors: list[Exception] = []

    def hold_maintenance_lock() -> None:
        try:
            with moves.reserve_gallery_maintenance():
                maintenance_entered.set()
                if not release_maintenance.wait(timeout=10):
                    maintenance_errors.append(TimeoutError("maintenance hold was not released"))
        except Exception as error:
            maintenance_errors.append(error)

    maintenance_thread = threading.Thread(target=hold_maintenance_lock)
    maintenance_thread.start()

    stop_event = threading.Event()
    resume_event = threading.Event()
    resume_event.set()
    poll_now_event = threading.Event()
    worker = _SessionWorker(device=None, runner=object())
    processor._invoker.services.logger = InvokeAILogger.get_logger()
    processor._polling_interval = 5
    processor._thread_semaphore = threading.BoundedSemaphore(1)
    processor._stop_event = stop_event
    processor._resume_event = resume_event
    processor._poll_now_event = poll_now_event
    processor._workers = [worker]

    claim_attempted = threading.Event()
    dequeue = processor._dequeue_if_storage_maintenance_inactive

    def observe_claim(device: str | None):
        claim_attempted.set()
        return dequeue(device)

    processor._dequeue_if_storage_maintenance_inactive = observe_claim  # type: ignore[method-assign]
    worker_thread = threading.Thread(
        target=processor._process,
        kwargs={
            "worker": worker,
            "stop_event": stop_event,
            "poll_now_event": poll_now_event,
            "resume_event": resume_event,
        },
        daemon=True,
    )
    stopped_before_release = False
    worker_started = False

    try:
        assert maintenance_entered.wait(timeout=10), "maintenance did not acquire the mutation lock"
        worker_thread.start()
        worker_started = True
        assert claim_attempted.wait(timeout=10), "worker did not attempt a queue claim"

        processor.stop()
        worker_thread.join(timeout=2)
        stopped_before_release = not worker_thread.is_alive()
    finally:
        processor.stop()
        release_maintenance.set()
        if worker_started:
            worker_thread.join(timeout=10)
        maintenance_thread.join(timeout=10)
        moves.stop()

    assert stopped_before_release, "worker shutdown waited for the gallery maintenance lock"
    assert not worker_thread.is_alive()
    assert not maintenance_thread.is_alive()
    assert maintenance_errors == []
    assert queue.dequeued == 0
