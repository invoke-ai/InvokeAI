import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from invokeai.app.invocations.remote_worker import model_transfer_state as transfer_state
from invokeai.app.invocations.remote_worker import remote_nodes as nodes


@pytest.fixture
def transfer_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer_state, "_TRANSFER_TASKS", {})
    monkeypatch.setattr(transfer_state, "_TRANSFER_SEQUENCE", 0)


def _queue(item_id: int) -> SimpleNamespace:
    return SimpleNamespace(item_id=item_id, user_id="alice", origin=f"webv2:queue-{item_id}")


def test_transfer_lock_is_shared_only_by_worker_and_hash(transfer_registry: None) -> None:
    first_id, first = transfer_state.register_model_transfer("http://worker-a", "hash-a")
    second_id, second = transfer_state.register_model_transfer("http://worker-a", "hash-a")
    other_worker_id, other_worker = transfer_state.register_model_transfer("http://worker-b", "hash-a")
    other_model_id, other_model = transfer_state.register_model_transfer("http://worker-a", "hash-b")

    assert first.shared_lock is second.shared_lock
    assert first.shared_lock is not other_worker.shared_lock
    assert first.shared_lock is not other_model.shared_lock

    for transfer_id in (first_id, second_id, other_worker_id, other_model_id):
        transfer_state.unregister_model_transfer(transfer_id)


def test_simultaneous_invokes_install_missing_model_once(
    transfer_registry: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"model")
    model = SimpleNamespace(path=path, name="model.gguf", hash="sha256:model")
    monkeypatch.setattr(nodes, "resolve_local_model_file", lambda services, identifier: model)
    monkeypatch.setattr(nodes, "_emit_model_transfer_progress", lambda **kwargs: None)

    class ModelServer:
        url = "http://127.0.0.1/model"

        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "ModelServer":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(nodes, "TemporaryModelServer", ModelServer)

    class Worker:
        config = SimpleNamespace(base_url="http://worker-a")

        def __init__(self) -> None:
            self.install_started = threading.Event()
            self.finish_install = threading.Event()
            self.installed = threading.Event()
            self.install_count = 0

        def get_model_by_hash(self, model_hash: str) -> dict[str, Any] | None:
            assert model_hash == model.hash
            return {"model": {"key": "remote-model"}} if self.installed.is_set() else None

        def install_model_from_url(self, url: str, *, name: str) -> dict[str, Any]:
            assert name == model.name
            self.install_count += 1
            self.install_started.set()
            return {"id": 15, "status": "waiting"}

        def get_model_install_job(self, job_id: int) -> dict[str, Any]:
            assert job_id == 15
            if not self.finish_install.wait(10):
                raise TimeoutError("Test model install did not finish")
            self.installed.set()
            return {"id": 15, "status": "completed", "bytes": 5, "total_bytes": 5}

    worker = Worker()
    services = SimpleNamespace(
        session_queue=SimpleNamespace(get_queue_item=lambda item_id: SimpleNamespace(status="in_progress"))
    )

    def run(item_id: int) -> None:
        context = SimpleNamespace(
            _services=services,
            _data=SimpleNamespace(queue_item=_queue(item_id)),
            logger=logging.getLogger("test-remote-model-transfer"),
        )
        nodes._transfer_missing_model_to_remote(
            context=context,
            invocation=SimpleNamespace(),
            remote_client=worker,
            remote_index=1,
            identifier={"key": "model"},
            transfer_host="",
            timeout_seconds=30,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 101)
        try:
            assert worker.install_started.wait(5)
            second = pool.submit(run, 102)
            deadline = time.monotonic() + 5
            registered = 0
            while time.monotonic() < deadline:
                with transfer_state._LOCK:
                    registered = len(transfer_state._TRANSFER_TASKS)
                if registered == 2:
                    break
                time.sleep(0.01)
            assert registered == 2
        finally:
            worker.finish_install.set()

        first.result(timeout=10)
        second.result(timeout=10)

    assert worker.install_count == 1
    assert transfer_state._TRANSFER_TASKS == {}
