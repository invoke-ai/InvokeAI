"""Remote Worker pool cleanup for successful, failed and canceled worker jobs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.remote_worker import remote_nodes, worker_pool
from invokeai.app.invocations.remote_worker.remote_client import RemoteInvokeClient, RemoteInvokeError


def test_delete_queue_item_uses_authenticated_transport_and_empty_response():
    client = object.__new__(RemoteInvokeClient)
    client._request = Mock(return_value=b"")
    client.delete_queue_item(42, "worker queue")
    client._request.assert_called_once_with("DELETE", "/api/v1/queue/worker%20queue/i/42")


@pytest.fixture
def pool_environment(monkeypatch):
    client = Mock()
    client.get_item.return_value = {"status": "completed"}
    client.delete_queue_item = Mock()

    queue_item = SimpleNamespace(
        item_id=321,
        session=SimpleNamespace(prepared_source_mapping={}, results={}),
    )
    invocation = SimpleNamespace(id="dispatch")
    persistence_order: list[str] = []
    services = SimpleNamespace(
        session_queue=SimpleNamespace(
            complete_queue_item=Mock(side_effect=lambda *_args, **_kwargs: persistence_order.append("complete")),
            fail_queue_item=Mock(),
            save_queue_item_session=Mock(side_effect=lambda *_args, **_kwargs: persistence_order.append("persist")),
        ),
        logger=SimpleNamespace(info=Mock(), warning=Mock(), error=Mock(), debug=Mock()),
    )
    settings = worker_pool.PoolSettings(
        mode="Distributed",
        workers=(),
        result_destination="gallery",
        local_gallery_board_id="",
        keep_remote_copies=False,
        auto_transfer_missing_models=False,
        model_transfer_host="",
        model_transfer_timeout_seconds=7200,
        poll_interval_seconds=0.25,
        timeout_seconds=60,
    )
    worker = worker_pool.WorkerSpec(url="http://worker.test", name="Remote 2", slot=2)
    local_status = ["in_progress"]

    monkeypatch.setattr(worker_pool, "_helper_for_queue_item", lambda _item: invocation)
    monkeypatch.setattr(
        worker_pool,
        "_dispatch_remote",
        lambda *_args, **_kwargs: (client, 42, "my-board", ["input.png"], ["input.mp4"]),
    )
    monkeypatch.setattr(worker_pool, "_emit_started", lambda *_args, **_kwargs: queue_item)
    monkeypatch.setattr(worker_pool, "_emit_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_pool, "_emit_result", lambda *_args, **_kwargs: None)
    importer = Mock(return_value=([], []))
    monkeypatch.setattr(worker_pool, "_import_completed", importer)
    monkeypatch.setattr(worker_pool, "_status", lambda *_args, **_kwargs: local_status[0])

    def run():
        return worker_pool._run_remote_job(
            services,
            queue_item,
            settings,
            worker,
            complete_local=True,
        )

    return SimpleNamespace(
        run=run,
        client=client,
        importer=importer,
        local_status=local_status,
        persistence_order=persistence_order,
        queue_item=queue_item,
        services=services,
        settings=settings,
        worker=worker,
    )


def test_worker_lane_retries_unavailable_worker_while_eligible_work_remains(monkeypatch):
    eligibility = iter([True, False, False, False, False])
    services = SimpleNamespace(logger=SimpleNamespace(warning=Mock()))
    pool = worker_pool._RemoteWorkerPool(services, "default", "user-1")
    worker = worker_pool.WorkerSpec(url="http://worker.test", name="Remote 1", slot=1)
    client = Mock()
    client.get_current_item.side_effect = RuntimeError("offline")
    lock = Mock()
    lock.acquire.return_value = True
    claim = Mock()

    monkeypatch.setattr(worker_pool, "_park_remote_only_items", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        worker_pool,
        "_has_eligible_item",
        lambda *_args, **_kwargs: next(eligibility),
    )
    monkeypatch.setattr(worker_pool, "_slot_lock", lambda _worker: lock)
    monkeypatch.setattr(remote_nodes, "_remote_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(worker_pool, "_claim_for_worker", claim)
    monkeypatch.setattr(worker_pool.time, "sleep", lambda _seconds: None)

    pool._run_lane(worker)

    client.get_current_item.assert_called_once()
    claim.assert_not_called()
    lock.release.assert_called_once()
    assert "will retry while eligible work remains" in services.logger.warning.call_args.args[0]


def test_success_imports_then_deletes_exact_remote_queue_item(pool_environment):
    env = pool_environment
    assert env.run() == "completed"
    env.importer.assert_called_once()
    env.client.delete_queue_item.assert_called_once_with(42, "default")
    env.services.session_queue.complete_queue_item.assert_called_once_with(321)
    assert env.services.session_queue.save_queue_item_session.call_count == 2
    assert env.services.session_queue.save_queue_item_session.call_args_list[0].args == (321, env.queue_item.session)
    assert env.services.session_queue.save_queue_item_session.call_args_list[1].args == (321, env.queue_item.session)
    assert env.persistence_order == ["persist", "complete", "persist"]


def test_result_history_persistence_failure_does_not_uncomplete_remote_item(pool_environment):
    env = pool_environment
    env.services.session_queue.save_queue_item_session.side_effect = RuntimeError("history write failed")

    assert env.run() == "completed"

    env.services.session_queue.complete_queue_item.assert_called_once_with(321)
    assert env.services.session_queue.save_queue_item_session.call_count == 2
    assert env.services.logger.warning.call_count == 2
    assert all("history write failed" in call.args[0] for call in env.services.logger.warning.call_args_list)


def test_emit_result_persists_remote_output_under_real_source_id():
    queue_session = SimpleNamespace(prepared_source_mapping={}, results={})
    event_session = SimpleNamespace(prepared_source_mapping={}, results={})
    queue_item = SimpleNamespace(session=queue_session)
    event_item = SimpleNamespace(session=event_session)
    synthetic_invocation = SimpleNamespace(id="synthetic")
    invocation = Mock()
    invocation.id = "dispatch"
    invocation.model_copy.return_value = synthetic_invocation
    output = object()
    services = SimpleNamespace(events=SimpleNamespace(emit_invocation_started=Mock(), emit_invocation_complete=Mock()))

    worker_pool._emit_result(
        services,
        queue_item,
        event_item,
        invocation,
        source_id="canvas_output",
        output=output,
    )

    assert queue_session.results == {"canvas_output": output}
    assert queue_session.prepared_source_mapping == {}

    event_id = next(iter(event_session.results))
    assert event_session.results[event_id] is output
    assert event_session.prepared_source_mapping[event_id] == "canvas_output"


def test_emit_result_keeps_multiple_outputs_without_fake_prepared_nodes():
    first = object()
    second = object()
    queue_session = SimpleNamespace(prepared_source_mapping={}, results={})
    event_session = SimpleNamespace(prepared_source_mapping={}, results={})
    queue_item = SimpleNamespace(session=queue_session)
    event_item = SimpleNamespace(session=event_session)
    invocation = Mock()
    invocation.id = "dispatch"
    invocation.model_copy.side_effect = lambda **kwargs: SimpleNamespace(id=kwargs["update"]["id"])
    services = SimpleNamespace(events=SimpleNamespace(emit_invocation_started=Mock(), emit_invocation_complete=Mock()))

    worker_pool._emit_result(
        services,
        queue_item,
        event_item,
        invocation,
        source_id="canvas_output",
        output=first,
    )
    worker_pool._emit_result(
        services,
        queue_item,
        event_item,
        invocation,
        source_id="canvas_output",
        output=second,
    )

    assert queue_session.results["canvas_output"] is second
    assert first in queue_session.results.values()
    assert queue_session.prepared_source_mapping == {}


def test_remote_media_source_ids_follow_remote_prepared_source_mapping():
    image_sources, video_sources = worker_pool._remote_media_source_ids(
        {
            "session": {
                "prepared_source_mapping": {
                    "image-exec": "canvas_output",
                    "video-exec": "video_output",
                },
                "results": {
                    "image-exec": {"image": {"image_name": "remote.png"}},
                    "video-exec": {"video": {"video_name": "remote.mp4"}},
                },
            }
        }
    )

    assert image_sources == {"remote.png": "canvas_output"}
    assert video_sources == {"remote.mp4": "video_output"}


def test_completed_remote_job_cleans_transferred_inputs(pool_environment, monkeypatch):
    env = pool_environment
    cleanup = Mock()
    monkeypatch.setattr(worker_pool, "_cleanup_remote_inputs", cleanup)

    assert env.run() == "completed"

    cleanup.assert_called_once_with(
        env.client,
        env.services,
        env.settings,
        ["input.png"],
        ["input.mp4"],
        reason="remote job completed",
    )


def test_cleanup_remote_inputs_deletes_each_transferred_input_once(pool_environment):
    env = pool_environment

    worker_pool._cleanup_remote_inputs(
        env.client,
        env.services,
        env.settings,
        ["input.png", "input.png"],
        ["input.mp4", "input.mp4"],
        reason="test cleanup",
    )

    env.client.delete_image.assert_called_once_with("input.png")
    env.client.delete_video.assert_called_once_with("input.mp4")


def test_cleanup_remote_inputs_preserves_inputs_when_keep_copies_enabled(pool_environment):
    env = pool_environment
    env.settings = worker_pool.PoolSettings(
        mode=env.settings.mode,
        workers=env.settings.workers,
        result_destination=env.settings.result_destination,
        local_gallery_board_id=env.settings.local_gallery_board_id,
        keep_remote_copies=True,
        auto_transfer_missing_models=env.settings.auto_transfer_missing_models,
        model_transfer_host=env.settings.model_transfer_host,
        model_transfer_timeout_seconds=env.settings.model_transfer_timeout_seconds,
        poll_interval_seconds=env.settings.poll_interval_seconds,
        timeout_seconds=env.settings.timeout_seconds,
    )

    worker_pool._cleanup_remote_inputs(
        env.client,
        env.services,
        env.settings,
        ["input.png"],
        ["input.mp4"],
        reason="test preserve",
    )

    env.client.delete_image.assert_not_called()
    env.client.delete_video.assert_not_called()


def test_failed_import_keeps_remote_queue_record(pool_environment):
    env = pool_environment
    env.importer.side_effect = RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        env.run()
    env.client.delete_queue_item.assert_not_called()
    env.services.session_queue.complete_queue_item.assert_not_called()


def test_failed_worker_item_keeps_remote_queue_record(pool_environment):
    env = pool_environment
    env.client.get_item.return_value = {"status": "failed", "session": {"errors": {"node": "oops"}}}
    with pytest.raises(RemoteInvokeError, match="failed"):
        env.run()
    env.importer.assert_not_called()
    env.client.delete_queue_item.assert_not_called()
    env.services.session_queue.fail_queue_item.assert_not_called()


def test_remote_oom_marks_local_item_failed_instead_of_requeueing(pool_environment):
    env = pool_environment
    env.client.get_item.return_value = {
        "status": "failed",
        "session": {
            "errors": {
                "node": (
                    "OutOfMemoryError: Allocation on device 0 would exceed allowed memory. "
                    "Free (according to CUDA): 32.50 MiB"
                )
            }
        },
    }

    with pytest.raises(RemoteInvokeError, match="OutOfMemoryError"):
        env.run()

    env.services.session_queue.fail_queue_item.assert_called_once()
    call = env.services.session_queue.fail_queue_item.call_args.kwargs
    assert call["item_id"] == 321
    assert call["error_type"] == "OutOfMemoryError"
    assert "Remote 2 ran out of memory" in call["error_message"]
    env.importer.assert_not_called()


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        ({"node": "torch.OutOfMemoryError: device allocation failed"}, True),
        ({"node": "CUDA out of memory. Tried to allocate 1.00 GiB"}, True),
        ({"node": "Allocation on device 0 would exceed allowed memory."}, True),
        ({"node": "worker disconnected"}, False),
        ({"node": "model not found"}, False),
    ],
)
def test_remote_oom_detection_is_narrow(errors, expected):
    assert worker_pool._is_remote_oom_error(errors) is expected


def test_queue_delete_error_does_not_change_completed_result(pool_environment):
    env = pool_environment
    env.client.delete_queue_item.side_effect = RuntimeError("worker disconnected")
    assert env.run() == "completed"
    env.services.logger.warning.assert_called_once()
    assert "worker disconnected" in env.services.logger.warning.call_args.args[0]


def test_native_queue_cancellation_cancels_remote_job(pool_environment, monkeypatch):
    env = pool_environment
    env.local_status[0] = "canceled"
    cancel = Mock()
    monkeypatch.setattr(worker_pool, "_cancel_remote", cancel)

    assert env.run() == "canceled"
    cancel.assert_called_once_with(env.client, 42, env.services, env.worker)
    env.importer.assert_not_called()


def test_cancel_remote_deletes_confirmed_canceled_worker_item(monkeypatch):
    client = Mock()
    client.get_item.return_value = {"status": "canceled"}
    services = SimpleNamespace(logger=SimpleNamespace(warning=Mock()))
    worker = worker_pool.WorkerSpec(url="http://worker.test", name="Remote 1", slot=1)

    worker_pool._cancel_remote(client, 42, services, worker)

    client.cancel_queue_item.assert_called_once_with(42, "default")
    client.delete_queue_item.assert_called_once_with(42, "default")
