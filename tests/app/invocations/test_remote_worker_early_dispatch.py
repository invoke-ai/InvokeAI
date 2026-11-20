"""Queue-time scheduling for automatic Remote Worker dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock, call

from invokeai.app.invocations.remote_worker import early_dispatch, worker_pool


def test_schedule_automatic_remote_dispatches_schedules_all_enqueued_items(monkeypatch):
    schedule = Mock()
    services = object()
    helper = SimpleNamespace(get_type=lambda: early_dispatch.AUTOMATIC_REMOTE_WORKER_NODE_TYPE)
    batch = SimpleNamespace(graph=SimpleNamespace(nodes={early_dispatch.AUTOMATIC_REMOTE_WORKER_NODE_ID: helper}))
    monkeypatch.setattr(worker_pool, "schedule_remote_worker_pool", schedule)

    scheduled = early_dispatch.schedule_automatic_remote_dispatches(
        batch=batch,
        item_ids=[11, 12],
        services=services,
    )

    assert scheduled is True
    assert schedule.call_args_list == [call(11, services), call(12, services)]


def test_schedule_automatic_remote_dispatches_ignores_normal_batches(monkeypatch):
    schedule = Mock()
    services = object()
    batch = SimpleNamespace(graph=SimpleNamespace(nodes={}))
    monkeypatch.setattr(worker_pool, "schedule_remote_worker_pool", schedule)

    scheduled = early_dispatch.schedule_automatic_remote_dispatches(
        batch=batch,
        item_ids=[11],
        services=services,
    )

    assert scheduled is False
    schedule.assert_not_called()


def _priority_claim_environment(monkeypatch, modes):
    """A small queue with real per-item settings and atomic-claim style semantics."""
    worker = worker_pool.WorkerSpec(url="http://worker.test", name="Remote 1", slot=1)
    items = {
        index: SimpleNamespace(item_id=index, status="pending", mode=mode) for index, mode in enumerate(modes, start=1)
    }
    queue = SimpleNamespace(
        get_root_queue_item_ids=Mock(return_value=list(items)),
        get_queue_item=Mock(side_effect=items.__getitem__),
        transition_queue_item_if_status=Mock(side_effect=lambda item_id, *_args, **_kwargs: items[item_id]),
    )
    services = SimpleNamespace(session_queue=queue)

    def settings_for(item):
        return SimpleNamespace(mode=item.mode, workers=(worker,))

    monkeypatch.setattr(worker_pool, "_settings_for_queue_item", settings_for)
    return services, worker, queue, items


def test_distributed_reserves_first_two_items_for_idle_local_gpus(monkeypatch):
    services, worker, queue, items = _priority_claim_environment(
        monkeypatch, ["Distributed", "Distributed", "Distributed", "Distributed"]
    )
    claimed = worker_pool._claim_for_worker(
        services, queue_id="default", user_id="user", worker=worker, excluded=set(), local_reservation=2
    )
    assert claimed[0] is items[3]
    queue.transition_queue_item_if_status.assert_called_once()
    assert queue.transition_queue_item_if_status.call_args.args[:3] == (3, "in_progress", ("pending",))


def test_distributed_claims_first_job_without_idle_local_gpu(monkeypatch):
    services, worker, queue, items = _priority_claim_environment(monkeypatch, ["Distributed", "Distributed"])
    claimed = worker_pool._claim_for_worker(
        services, queue_id="default", user_id="user", worker=worker, excluded=set(), local_reservation=0
    )
    assert claimed[0] is items[1]


def test_distributed_waits_if_all_pending_items_are_reserved_for_local(monkeypatch):
    services, worker, queue, _items = _priority_claim_environment(monkeypatch, ["Distributed", "Distributed"])
    assert (
        worker_pool._claim_for_worker(
            services, queue_id="default", user_id="user", worker=worker, excluded=set(), local_reservation=2
        )
        is None
    )
    queue.transition_queue_item_if_status.assert_not_called()


def test_remote_only_ignores_local_gpu_reservation(monkeypatch):
    services, worker, queue, items = _priority_claim_environment(monkeypatch, ["Distributed", "Remote Only"])
    claimed = worker_pool._claim_for_worker(
        services, queue_id="default", user_id="user", worker=worker, excluded=set(), local_reservation=2
    )
    assert claimed[0] is items[2]
    queue.transition_queue_item_if_status.assert_called_once()


def test_local_reservation_expires_and_does_not_block_busy_local_gpus(monkeypatch):
    clock = iter([10.0, 10.5, 11.1])
    monkeypatch.setattr(worker_pool.time, "monotonic", lambda: next(clock))
    services = SimpleNamespace(session_processor=SimpleNamespace(get_available_local_worker_count=lambda: 2))
    count, first = worker_pool._local_worker_reservation(services, None)
    assert (count, first) == (2, 10.0)
    assert worker_pool._local_worker_reservation(services, first) == (2, first)
    assert worker_pool._local_worker_reservation(services, first) == (0, first)
    services.session_processor.get_available_local_worker_count = lambda: 0
    assert worker_pool._local_worker_reservation(services, first) == (0, None)


def test_local_priority_ignores_paused_or_lent_gpu_workers(monkeypatch):
    from threading import Event

    from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor

    processor = object.__new__(DefaultSessionProcessor)
    processor._resume_event = Event()
    processor._stop_event = Event()
    processor._resume_event.set()
    processor._invoker = SimpleNamespace(services=SimpleNamespace(image_moves=None))
    processor._workers = [
        SimpleNamespace(queue_item=None, thread=SimpleNamespace(is_alive=lambda: True), device="cuda:0"),
        SimpleNamespace(queue_item=None, thread=SimpleNamespace(is_alive=lambda: True), device="cuda:1"),
    ]

    import invokeai.app.services.session_processor.session_processor_default as processor_module

    monkeypatch.setattr(processor_module.GENERATION_DEVICE_POOL, "is_lent", lambda dev: dev == "cuda:1")
    assert processor.get_available_local_worker_count() == 1
    processor._workers[0].queue_item = SimpleNamespace(item_id=42)
    assert processor.get_available_local_worker_count() == 0
    processor._workers[0].queue_item = None
    processor._resume_event.clear()
    assert processor.get_available_local_worker_count() == 0
