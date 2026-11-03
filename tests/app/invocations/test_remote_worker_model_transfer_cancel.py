"""Focused regression coverage for Remote Worker model-transfer cancellation."""

import threading
from types import SimpleNamespace

import pytest

from invokeai.app.invocations.remote_worker import diffusers_transfer as receiver
from invokeai.app.invocations.remote_worker import model_transfer_state as transfer_state
from invokeai.app.invocations.remote_worker import remote_nodes as nodes


@pytest.fixture
def transfer_registry(monkeypatch):
    monkeypatch.setattr(transfer_state, "_TRANSFER_TASKS", {})
    monkeypatch.setattr(transfer_state, "_TRANSFER_SEQUENCE", 0)


def test_other_generation_keeps_shared_directory_alive(transfer_registry):
    first_id, first = transfer_state.register_model_transfer("http://worker", "hash")
    second_id, second = transfer_state.register_model_transfer("http://worker", "hash")
    first.cancel_requested.set()
    assert transfer_state.another_generation_needs_model(first)
    second.cancel_requested.set()
    assert not transfer_state.another_generation_needs_model(first)
    transfer_state.unregister_model_transfer(first_id)
    transfer_state.unregister_model_transfer(second_id)


def test_other_generation_keeps_shared_preparation_alive(transfer_registry):
    first_id, first = transfer_state.register_model_transfer("http://worker", "hash")
    second_id, second = transfer_state.register_model_transfer("http://worker", "hash")
    first.cancel_requested.set()

    assert not (first.cancel_requested.is_set() and not transfer_state.another_generation_needs_model(first))

    second.cancel_requested.set()
    assert first.cancel_requested.is_set() and not transfer_state.another_generation_needs_model(first)

    transfer_state.unregister_model_transfer(first_id)
    transfer_state.unregister_model_transfer(second_id)


@pytest.mark.parametrize("directory", [False, True])
def test_native_queue_cancellation_stops_worker_install(transfer_registry, monkeypatch, tmp_path, directory):
    source = tmp_path / ("directory_model" if directory else "model.safetensors")
    if directory:
        source.mkdir()
        (source / "model_index.json").write_text("{}")
    else:
        source.write_bytes(b"checkpoint")

    model = SimpleNamespace(path=source, hash="hash", name="model")
    queue_item = SimpleNamespace(item_id=298108, user_id="alice", origin="webv2:queue", status="in_progress")
    services = SimpleNamespace(session_queue=SimpleNamespace(get_queue_item=lambda _: queue_item))
    context = SimpleNamespace(
        _data=SimpleNamespace(queue_item=queue_item),
        _services=services,
        logger=SimpleNamespace(warning=lambda *args: None, info=lambda *args: None, debug=lambda *args: None),
    )
    events = []

    monkeypatch.setattr(nodes, "resolve_local_model_file", lambda *_: model)
    monkeypatch.setattr(nodes, "_emit_model_transfer_progress", lambda **kwargs: events.append(kwargs["phase"]))

    class TempServer:
        url = "http://primary/token/model.safetensors"

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class TempDirectoryServer(TempServer):
        def __init__(self, **kwargs):
            self.files = [SimpleNamespace(size=0)]

        def manifest(self, **kwargs):
            return {"name": "model"}

    monkeypatch.setattr(nodes, "TemporaryModelServer", TempServer)
    monkeypatch.setattr(nodes, "TemporaryDirectoryModelServer", TempDirectoryServer)

    class Client:
        config = SimpleNamespace(base_url="http://worker")

        def __init__(self):
            self.deleted = []
            self.polls = 0

        def get_model_by_hash(self, _):
            return None

        def list_models(self):
            return []

        def install_model_from_url(self, *_args, **_kwargs):
            return {"id": 14, "status": "downloading"}

        def install_directory_from_manifest(self, *_args, **_kwargs):
            return {"id": 14, "status": "downloading"}

        def get_directory_install_job(self, job_id):
            return self.get_model_install_job(job_id)

        def get_model_install_job(self, _):
            self.polls += 1
            if self.polls == 1:
                queue_item.status = "canceled"
            return {"status": "cancelled" if self.deleted else "downloading"}

        def _request(self, method, path):
            self.deleted.append((method, path))

    client = Client()
    with pytest.raises(nodes.RemoteModelTransferCancelled):
        nodes._transfer_missing_model_to_remote(
            context=context,
            invocation=object(),
            remote_client=client,
            remote_index=2,
            identifier={"key": "key"},
            transfer_host="",
            timeout_seconds=60,
        )

    path = "/api/v1/remote_workers/diffusers/install/14" if directory else "/api/v2/models/install/14"
    assert client.deleted == [("DELETE", path)]
    assert events[-1] == "cancelled"
    assert transfer_state._TRANSFER_TASKS == {}


def test_receiver_cancel_signals_only_live_job(monkeypatch):
    monkeypatch.setattr(
        receiver, "_JOBS", {12: {"id": 12, "status": "downloading"}, 13: {"id": 13, "status": "completed"}}
    )
    events = {12: threading.Event(), 13: threading.Event()}
    monkeypatch.setattr(receiver, "_CANCEL_EVENTS", events)
    assert receiver.cancel_directory_install_job(12)["status"] == "downloading"
    assert events[12].is_set()
    assert receiver.cancel_directory_install_job(13)["status"] == "completed"
    assert not events[13].is_set()
    assert receiver.cancel_directory_install_job(999) is None


def test_receiver_cancel_tolerates_cleanup_window_without_event(monkeypatch):
    monkeypatch.setattr(receiver, "_JOBS", {12: {"id": 12, "status": "downloading"}})
    monkeypatch.setattr(receiver, "_CANCEL_EVENTS", {})

    assert receiver.cancel_directory_install_job(12) == {"id": 12, "status": "downloading"}


def test_receiver_cancelled_job_finalization_releases_state_atomically(monkeypatch):
    job_id = 12
    model_hash = "hash"
    event = threading.Event()
    event.set()
    monkeypatch.setattr(
        receiver,
        "_JOBS",
        {job_id: {"id": job_id, "status": "waiting", "bytes": 0, "total_bytes": 1, "error": None}},
    )
    monkeypatch.setattr(receiver, "_CANCEL_EVENTS", {job_id: event})
    monkeypatch.setattr(receiver, "_ACTIVE_HASHES", {model_hash: job_id})

    services = SimpleNamespace(logger=SimpleNamespace(error=lambda *_args, **_kwargs: None))
    manifest = ("http://worker/token", "model", model_hash, [], 1)

    receiver._run_install(job_id, manifest, services)

    assert receiver._JOBS[job_id]["status"] == "cancelled"
    assert model_hash not in receiver._ACTIVE_HASHES
    assert job_id not in receiver._CANCEL_EVENTS
    assert receiver.cancel_directory_install_job(job_id)["status"] == "cancelled"


def test_receiver_refuses_to_reuse_cancelling_install(monkeypatch):
    monkeypatch.setattr(receiver, "_JOBS", {12: {"id": 12, "status": "downloading"}})
    monkeypatch.setattr(receiver, "_ACTIVE_HASHES", {"hash": 12})
    event = threading.Event()
    event.set()
    monkeypatch.setattr(receiver, "_CANCEL_EVENTS", {12: event})
    monkeypatch.setattr(receiver, "_validated_manifest", lambda _: ("http://worker/token", "model", "hash", [], 1))
    with pytest.raises(ValueError, match="cancellation is still being cleaned up"):
        receiver.start_directory_install({}, object())
