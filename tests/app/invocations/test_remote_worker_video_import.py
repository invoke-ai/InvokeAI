"""Remote Worker final video imports through the backend worker pool."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.remote_worker import remote_media, remote_nodes, worker_pool
from invokeai.app.invocations.remote_worker.remote_client import RemoteInvokeClient


def test_extract_video_names_from_native_and_list_results():
    item = {
        "session": {
            "results": {
                "video": {"video": {"video_name": "one.mp4"}},
                "batch": {"videos": [{"video_name": "two.mp4"}, {"video_name": "one.mp4"}]},
                "image": {"image": {"image_name": "unrelated.png"}},
            }
        }
    }
    assert RemoteInvokeClient.extract_video_names(item) == ["one.mp4", "two.mp4"]
    assert RemoteInvokeClient.extract_image_names(item, allow_empty=True) == ["unrelated.png"]
    assert (
        RemoteInvokeClient.extract_image_names(
            {"session": {"results": {"video": {"video": {"video_name": "one.mp4"}}}}}, allow_empty=True
        )
        == []
    )


def test_gallery_filter_only_non_intermediate_videos():
    client = object.__new__(RemoteInvokeClient)
    client.get_video_dto = Mock(side_effect=[{"is_intermediate": True}, {"is_intermediate": False}])
    assert client.filter_gallery_video_names(["draft.mp4", "final.mp4"]) == ["final.mp4"]


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_media, "probe_video_with_codec", lambda _path: (640, 480, 1.5, 24.0, "h264"))
    saved = []

    def create_video(**kwargs):
        saved.append((kwargs, Path(kwargs["source_path"]).read_bytes()))
        return SimpleNamespace(video_name="primary.mp4")

    services = SimpleNamespace(
        configuration=SimpleNamespace(multiuser=False, outputs_path=tmp_path),
        videos=SimpleNamespace(create=Mock(side_effect=create_video)),
        images=SimpleNamespace(create=Mock()),
        logger=SimpleNamespace(info=Mock(), warning=Mock()),
    )
    client = Mock()
    client.extract_image_names.return_value = []
    client.extract_video_names.return_value = ["remote.mp4"]
    client.filter_gallery_image_names.return_value = []
    client.filter_gallery_video_names.return_value = ["remote.mp4"]
    client.download_video.return_value = b"mp4 fixture bytes"
    client.get_image_metadata.return_value = None
    client.get_video_metadata.return_value = None
    queue_item = SimpleNamespace(user_id="owner", workflow=None, session=None, session_id="session")
    invocation = SimpleNamespace(id="node")
    return services, client, queue_item, invocation, saved, tmp_path


def import_video(environment, *, keep=False):
    services, client, queue_item, invocation, *_ = environment
    settings = worker_pool.PoolSettings(
        mode="Distributed",
        workers=(),
        result_destination="gallery",
        local_gallery_board_id="board-1",
        keep_remote_copies=keep,
        auto_transfer_missing_models=False,
        model_transfer_host="",
        model_transfer_timeout_seconds=7200,
        poll_interval_seconds=0.75,
        timeout_seconds=14400,
    )
    images, videos = worker_pool._import_completed(
        services,
        queue_item,
        invocation,
        settings,
        client,
        {"session": {"results": {}}},
        "board-1",
    )
    return [*images, *videos]


def test_video_only_import_uses_configured_outputs_and_native_service(environment):
    services, client, _item, _invocation, saved, output_path = environment
    assert import_video(environment) == [SimpleNamespace(video_name="primary.mp4")]
    assert len(saved) == 1
    args, data = saved[0]
    assert data == b"mp4 fixture bytes"
    assert args["source_path"].parent == output_path / "videos"
    assert args["width"] == 640 and args["duration"] == 1.5 and args["fps"] == 24
    assert args["board_id"] == "board-1" and args["user_id"] == "owner"
    assert args["is_intermediate"] is False
    assert list((output_path / "videos").glob(".irw_remote_*")) == []
    client.delete_video.assert_called_once_with("remote.mp4")
    services.images.create.assert_not_called()


def test_keep_copies_preserves_remote_video(environment):
    _services, client, *_rest = environment
    import_video(environment, keep=True)
    client.delete_video.assert_not_called()


def test_mixed_image_video_import_retains_image_behavior(environment):
    services, client, *_rest = environment
    client.extract_image_names.return_value = ["remote.png"]
    client.filter_gallery_image_names.return_value = ["remote.png"]
    services.images.create.return_value = SimpleNamespace(image_name="primary.png")
    imported = import_video(environment)
    assert [getattr(dto, "image_name", None) or getattr(dto, "video_name", None) for dto in imported] == [
        "primary.png",
        "primary.mp4",
    ]
    services.images.create.assert_called_once()
    client.delete_image.assert_called_once_with("remote.png")
    client.delete_video.assert_called_once_with("remote.mp4")


def test_upload_input_video_streams_multipart_and_returns_remote_name(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video-bytes")

    client = object.__new__(RemoteInvokeClient)
    client._request = Mock(return_value=b'{"video_name":"remote-input.mp4"}')

    assert client.upload_input_video(source) == "remote-input.mp4"

    call = client._request.call_args
    assert call.args[0] == "POST"
    assert call.args[1] == "/api/v1/videos/upload?video_category=user&is_intermediate=true"
    assert call.kwargs["content_type"].startswith("multipart/form-data; boundary=irw-")
    assert call.kwargs["content_length"] > source.stat().st_size
    body = b"".join(call.kwargs["body"])
    assert b'filename="remote-input.mp4"' in body
    assert b"video-bytes" in body


def test_transfer_source_videos_uploads_once_and_remaps_all_references(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    context = SimpleNamespace(
        videos=SimpleNamespace(get_path=Mock(return_value=source)),
        logger=SimpleNamespace(debug=Mock(), info=Mock()),
    )
    client = SimpleNamespace(upload_input_video=Mock(return_value="remote.mp4"))
    graph = {
        "nodes": {
            "source": {"video": {"video_name": "local.mp4"}},
            "nested": {"clips": [{"video_name": "local.mp4"}]},
        }
    }

    uploaded_names: list[str] = []
    remote_nodes._transfer_source_videos_to_remote(
        context=context,
        remote_client=client,
        graph=graph,
        video_names=["local.mp4"],
        remote_index=1,
        uploaded_names=uploaded_names,
    )

    context.videos.get_path.assert_called_once_with("local.mp4")
    client.upload_input_video.assert_called_once_with(source)
    assert uploaded_names == ["remote.mp4"]
    assert graph["nodes"]["source"]["video"]["video_name"] == "remote.mp4"
    assert graph["nodes"]["nested"]["clips"][0]["video_name"] == "remote.mp4"


def test_video_client_uses_existing_authenticated_transport():
    client = object.__new__(RemoteInvokeClient)
    client._request = Mock(return_value=b"mp4")
    client._request_json = Mock(return_value={"is_intermediate": False})
    assert client.download_video("a b.mp4") == b"mp4"
    client._request.assert_called_once_with("GET", "/api/v1/videos/i/a%20b.mp4/full")
    client.get_video_dto("a b.mp4")
    client._request_json.assert_called_with("GET", "/api/v1/videos/i/a%20b.mp4")
    client.delete_video("a b.mp4")
    client._request_json.assert_called_with("DELETE", "/api/v1/videos/i/a%20b.mp4")


def test_failed_import_keeps_remote_video_and_cleans_staging(environment):
    services, client, _item, _invocation, _saved, output_path = environment
    services.videos.create.side_effect = RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        import_video(environment)
    client.delete_video.assert_not_called()
    assert list((output_path / "videos").glob(".irw_remote_*")) == []


def test_remote_metadata_client_preserves_json_objects_and_null():
    import json

    client = object.__new__(RemoteInvokeClient)
    client._request_json_value = Mock(
        side_effect=[
            {"prompt": "rabbit", "seed": 42, "unicode": "🐰"},
            {"prompt": "video", "fps": 24},
            None,
            None,
        ]
    )
    assert json.loads(client.get_image_metadata("a b.png")) == {"prompt": "rabbit", "seed": 42, "unicode": "🐰"}
    assert json.loads(client.get_video_metadata("x y.mp4")) == {"prompt": "video", "fps": 24}
    assert client.get_image_metadata("blank.png") is None
    assert client.get_video_metadata("blank.mp4") is None
    assert client._request_json_value.call_args_list[0].args == ("GET", "/api/v1/images/i/a%20b.png/metadata")
    assert client._request_json_value.call_args_list[1].args == ("GET", "/api/v1/videos/i/x%20y.mp4/metadata")


def test_remote_source_node_id_is_preserved_on_local_image_record(environment):
    services, client, queue_item, invocation, *_ = environment
    client.extract_image_names.return_value = ["remote.png"]
    client.extract_video_names.return_value = []
    client.filter_gallery_image_names.return_value = ["remote.png"]
    client.filter_gallery_video_names.return_value = []
    services.images.create.return_value = SimpleNamespace(image_name="primary.png")

    settings = worker_pool.PoolSettings(
        mode="Distributed",
        workers=(),
        result_destination="gallery",
        local_gallery_board_id="",
        keep_remote_copies=True,
        auto_transfer_missing_models=False,
        model_transfer_host="",
        model_transfer_timeout_seconds=7200,
        poll_interval_seconds=0.75,
        timeout_seconds=14400,
    )
    worker_pool._import_completed(
        services,
        queue_item,
        invocation,
        settings,
        client,
        {
            "session": {
                "prepared_source_mapping": {"image-exec": "canvas_output"},
                "results": {"image-exec": {"image": {"image_name": "remote.png"}}},
            }
        },
        "",
    )

    assert services.images.create.call_args.kwargs["node_id"] == "canvas_output"


def test_remote_image_and_video_metadata_reaches_local_services(environment):
    import json

    services, client, *_ = environment
    client.extract_image_names.return_value = ["remote.png"]
    client.filter_gallery_image_names.return_value = ["remote.png"]
    services.images.create.return_value = SimpleNamespace(image_name="primary.png")
    client.get_image_metadata.return_value = json.dumps({"prompt": "image", "seed": 15})
    client.get_video_metadata.return_value = json.dumps({"prompt": "video", "seed": 27})
    import_video(environment, keep=True)
    assert json.loads(services.images.create.call_args.kwargs["metadata"]) == {"prompt": "image", "seed": 15}
    assert json.loads(services.videos.create.call_args.kwargs["metadata"]) == {"prompt": "video", "seed": 27}
    client.delete_image.assert_not_called()
    client.delete_video.assert_not_called()


def test_failed_metadata_lookup_does_not_delete_remote_media(environment):
    _services, client, *_ = environment
    client.get_video_metadata.side_effect = RuntimeError("metadata unavailable")
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        import_video(environment)
    client.download_video.assert_not_called()
    client.delete_video.assert_not_called()
