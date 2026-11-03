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
