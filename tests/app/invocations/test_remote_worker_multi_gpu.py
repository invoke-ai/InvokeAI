"""Native multi-GPU remote worker capacity, URL-wide leases and compatibility."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.remote_worker import worker_pool
from invokeai.app.invocations.remote_worker.remote_client import RemoteConfig, RemoteInvokeClient, RemoteInvokeError
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor


def _worker(url: str = "http://remote.test", name: str = "Remote 1") -> worker_pool.WorkerSpec:
    return worker_pool.WorkerSpec(url=url, name=name, slot=1)


def test_remote_slot_limiter_allows_exactly_two_concurrent_jobs() -> None:
    slots = worker_pool._RemoteHostSlots()
    slots.set_capacity(2)
    assert slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    assert not slots.acquire(blocking=False)
    slots.release()
    assert slots.acquire(blocking=False)
    slots.release()
    slots.release()


def test_slot_limit_shrink_does_not_cancel_active_jobs() -> None:
    slots = worker_pool._RemoteHostSlots()
    slots.set_capacity(2)
    assert slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    slots.set_capacity(1)
    assert not slots.acquire(blocking=False)
    slots.release()
    assert not slots.acquire(blocking=False)
    slots.release()
    assert slots.acquire(blocking=False)
    slots.release()


def test_url_limiter_is_shared_across_user_pools_and_worker_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_pool, "_SLOT_LOCKS", {})
    first = worker_pool._slot_lock(_worker("http://remote.test/", "One"))
    second = worker_pool._slot_lock(_worker("http://REMOTE.test", "Two"))
    other = worker_pool._slot_lock(_worker("http://other.test", "Three"))
    assert first is second
    assert first is not other
    first.set_capacity(2)
    assert second.acquire(blocking=False)
    assert first.acquire(blocking=False)
    assert not second.acquire(blocking=False)
    second.release()
    first.release()


def test_pool_creates_one_lane_per_remote_generation_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    created = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(worker_pool.threading, "Thread", FakeThread)
    monkeypatch.setattr(worker_pool, "_SLOT_LOCKS", {})
    probe = Mock(return_value=2)
    monkeypatch.setattr(worker_pool, "_configured_remote_capacity", probe)
    pool = worker_pool._RemoteWorkerPool(services=object(), queue_id="default", user_id="alice")
    worker = _worker()
    pool.ensure_workers((worker,))
    pool.ensure_workers((worker,))
    assert len(pool._lanes) == 2
    assert len(created) == 2
    assert all(t.started for t in created)
    assert len({t.kwargs["name"] for t in created}) == 2
    probe.assert_called_once_with(worker, "alice")


def test_capacity_drop_does_not_spawn_extra_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(worker_pool.threading, "Thread", FakeThread)
    monkeypatch.setattr(worker_pool, "_SLOT_LOCKS", {})
    probe = Mock(side_effect=[2, 1])
    monkeypatch.setattr(worker_pool, "_configured_remote_capacity", probe)
    pool = worker_pool._RemoteWorkerPool(services=object(), queue_id="default", user_id="alice")
    worker = _worker()
    pool.ensure_workers((worker,))
    pool._capacity_cache.clear()
    pool.ensure_workers((worker,))
    slots = worker_pool._slot_lock(worker)
    assert len(pool._lanes) == 2  # Old lane exits when idle, while permits enforce the new limit.
    assert slots.acquire(blocking=False)
    assert not slots.acquire(blocking=False)
    slots.release()


def test_client_capacity_probe_and_older_worker_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RemoteInvokeClient(RemoteConfig(base_url="http://remote.test", api_key=""))
    request = Mock(return_value={"generation_workers": 2})
    monkeypatch.setattr(client, "_request_json", request)
    assert client.get_generation_worker_capacity() == 2
    request.assert_called_once_with("GET", "/api/v1/remote_workers/capacity")
    request.side_effect = RemoteInvokeError("Remote InvokeAI returned HTTP 404 for /api/v1/remote_workers/capacity")
    assert client.get_generation_worker_capacity() == 1
    request.side_effect = RemoteInvokeError("Remote InvokeAI returned HTTP 401 for /api/v1/remote_workers/capacity")
    with pytest.raises(RemoteInvokeError, match="401"):
        client.get_generation_worker_capacity()


@pytest.mark.parametrize("invalid", [0, -1, 17, "2", True, None])
def test_client_rejects_invalid_worker_capacity(monkeypatch: pytest.MonkeyPatch, invalid: object) -> None:
    client = RemoteInvokeClient(RemoteConfig(base_url="http://remote.test", api_key=""))
    monkeypatch.setattr(client, "_request_json", lambda *_args: {"generation_workers": invalid})
    with pytest.raises(RemoteInvokeError, match="capacity"):
        client.get_generation_worker_capacity()


def test_processor_reports_configured_workers_not_idle_workers() -> None:
    processor = object.__new__(DefaultSessionProcessor)
    processor._workers = []
    processor._devices = ["cuda:0", "cuda:1"]
    assert processor.get_generation_worker_capacity() == 2
    processor._workers = [SimpleNamespace(queue_item=object()), SimpleNamespace(queue_item=None)]
    assert processor.get_generation_worker_capacity() == 2
    processor._devices = None
    processor._workers = []
    assert processor.get_generation_worker_capacity() == 1


def test_two_gpu_pool_runs_two_remote_jobs_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from invokeai.app.invocations.remote_worker import remote_nodes

    monkeypatch.setattr(worker_pool, "_SLOT_LOCKS", {})
    monkeypatch.setattr(worker_pool, "_configured_remote_capacity", lambda *_args: 2)
    monkeypatch.setattr(worker_pool, "_park_remote_only_items", lambda *_args: None)
    monkeypatch.setattr(worker_pool, "_local_worker_reservation", lambda *_args: (0, None))
    monkeypatch.setattr(remote_nodes, "_remote_client", lambda *_args: SimpleNamespace(get_current_item=lambda: None))

    pending = [SimpleNamespace(item_id=1), SimpleNamespace(item_id=2)]
    guard = threading.Lock()
    started = []
    both_started = threading.Event()
    release = threading.Event()

    def has_work(*_args, **_kwargs):
        with guard:
            return bool(pending)

    def claim(*_args, **_kwargs):
        with guard:
            if not pending:
                return None
            return pending.pop(0), SimpleNamespace()

    def run_job(_services, queue_item, *_args, **_kwargs):
        with guard:
            started.append(queue_item.item_id)
            if len(started) == 2:
                both_started.set()
        if not release.wait(5):
            raise TimeoutError("Test jobs were never released")
        return "completed"

    monkeypatch.setattr(worker_pool, "_has_eligible_item", has_work)
    monkeypatch.setattr(worker_pool, "_claim_for_worker", claim)
    monkeypatch.setattr(worker_pool, "_run_remote_job", run_job)

    pool = worker_pool._RemoteWorkerPool(services=SimpleNamespace(logger=Mock()), queue_id="default", user_id="alice")
    pool.ensure_workers((_worker(),))
    threads = list(pool._lanes.values())
    try:
        assert both_started.wait(4), "Both remote jobs must start without waiting for one another"
        assert set(started) == {1, 2}
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
