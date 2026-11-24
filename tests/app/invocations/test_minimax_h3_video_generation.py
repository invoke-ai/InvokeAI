import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import imageio_ffmpeg
import pytest
from PIL import Image

from invokeai.app.invocations.fields import BoardField, ImageField, MetadataField
from invokeai.app.invocations.minimax_h3 import minimax_h3_video_generation as hosted
from invokeai.app.services.external_generation import minimax_video
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import (
    ImagesInterface,
    InvocationContextData,
    UtilInterface,
    build_invocation_context,
)
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.video_files.video_files_disk import DiskVideoFileStorage
from invokeai.app.services.videos.videos_default import VideoService
from invokeai.app.util.video_ingest import MediaProbe, probe_media_streams


@pytest.fixture(autouse=True)
def isolate_model_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing the in-memory database applies legacy migrations that also inspect the models directory.
    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))


@pytest.fixture
def invoker(mock_invoker: Invoker, tmp_path: Path) -> Invoker:
    services = mock_invoker.services
    services.configuration.multiuser = True
    services.configuration.external_minimax_api_key = "test-only-key"
    services.names = SimpleNameService()
    services.urls = LocalUrlService()
    services.image_files = DiskImageFileStorage(tmp_path / "images")
    services.image_files.start(mock_invoker)
    services.video_files = DiskVideoFileStorage(tmp_path / "videos")
    services.video_files.start(mock_invoker)
    services.videos = VideoService()
    services.videos.start(mock_invoker)
    return mock_invoker


@pytest.fixture
def owner(invoker: Invoker) -> str:
    return invoker.services.users.create(UserCreateRequest(email="owner@example.com", password="Test-Pass-135")).user_id


def _run(invoker: Invoker, node: hosted.MiniMaxH3VideoGenerationInvocation, user_id: str) -> hosted.VideoOutput:
    session = GraphExecutionState(graph=Graph(nodes={node.id: node}))
    item = SessionQueueItem(
        item_id=1,
        batch_id="batch",
        queue_id="default",
        session_id=session.id,
        session=session,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        started_at=None,
        completed_at=None,
        user_id=user_id,
    )
    prepared = session.next()
    assert isinstance(prepared, hosted.MiniMaxH3VideoGenerationInvocation)
    context = build_invocation_context(
        services=invoker.services,
        data=InvocationContextData(queue_item=item, invocation=prepared, source_invocation_id=node.id),
        is_canceled=lambda: False,
    )
    result = prepared.invoke(context)
    session.complete(prepared.id, result)
    return result


def _image(invoker: Invoker, owner: str, color: str) -> ImageField:
    image = invoker.services.images.create(
        image=Image.new("RGB", (320, 256), color),
        image_origin=ResourceOrigin.EXTERNAL,
        image_category=ImageCategory.GENERAL,
        user_id=owner,
    )
    return ImageField(image_name=image.image_name)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("codec", ["libx264", "mpeg4"])
def test_generated_video_reaches_owners_gallery_with_audio_thumbnail_and_metadata(
    invoker: Invoker, owner: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codec: str
) -> None:
    source = tmp_path / "provider.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=8:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:d=1.5",
            "-c:v",
            codec,
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        timeout=30,
    )
    board = invoker.services.board_records.save("Hosted videos", owner)
    node = hosted.MiniMaxH3VideoGenerationInvocation(
        prompt="Follow the bicycle",
        board=BoardField(board_id=board.board_id),
        first_frame=_image(invoker, owner, "red"),
        last_frame=_image(invoker, owner, "blue"),
        metadata=MetadataField(root={"credit": "studio"}),
    )

    @contextmanager
    def submit(*, api_key: str, request: hosted.MiniMaxVideoRequest, **kwargs) -> Iterator[Path]:
        assert api_key == "test-only-key"
        assert request.model == "MiniMax-H3"
        assert request.prompt == "Follow the bicycle"
        with request.first_frame() as first, request.last_frame() as last:
            assert first.getpixel((0, 0)) == (255, 0, 0)
            assert last.getpixel((0, 0)) == (0, 0, 255)
        assert kwargs["is_canceled"]() is False
        kwargs["on_progress"]("running")
        yield source

    monkeypatch.setattr(hosted, "generate_minimax_video", submit)
    output = _run(invoker, node, owner)
    name = output.video.video_name
    assert (output.width, output.height, output.num_frames, output.fps, output.duration) == (64, 48, 8, 8.0, 1.5)
    saved = invoker.services.videos.get_dto(name)
    assert invoker.services.video_records.get_user_id(name) == owner
    assert saved.board_id == board.board_id
    assert saved.has_workflow
    assert invoker.services.gallery.get_item_names(user_id=owner, board_id=board.board_id).item_names == [name]
    assert probe_media_streams(Path(invoker.services.videos.get_path(name))) == MediaProbe("h264", "aac")
    with Image.open(invoker.services.videos.get_path(name, thumbnail=True)) as thumb:
        assert thumb.format == "WEBP"
    assert invoker.services.videos.get_metadata(name).root == {
        "credit": "studio",
        "external_provider": "minimax",
        "external_model_id": "MiniMax-H3",
        "positive_prompt": "Follow the bicycle",
        "width": 64,
        "height": 48,
        "fps": 8,
        "num_frames": 8,
    }


@pytest.mark.parametrize("private_resource", ["board", "first_frame", "reference_images"])
def test_inaccessible_gallery_input_or_output_is_rejected_before_billing(
    invoker: Invoker, owner: str, monkeypatch: pytest.MonkeyPatch, private_resource: str
) -> None:
    other = invoker.services.users.create(UserCreateRequest(email="other@example.com", password="Test-Pass-246"))
    if private_resource == "board":
        private = BoardField(board_id=invoker.services.board_records.save("Private", other.user_id).board_id)
    else:
        image = _image(invoker, other.user_id, "red")
        private = [image] if private_resource == "reference_images" else image
    node = hosted.MiniMaxH3VideoGenerationInvocation(prompt="animate", **{private_resource: private})
    monkeypatch.setattr(minimax_video, "_submit", lambda *args: pytest.fail("Submitted before authorization"))
    with pytest.raises(PermissionError, match="not authorized"):
        _run(invoker, node, owner)


def test_non_video_result_never_reaches_the_gallery(
    invoker: Invoker, owner: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "returned.mp4"
    source.write_bytes(b"<html>unavailable</html>")

    @contextmanager
    def submit(**kwargs) -> Iterator[Path]:
        yield source

    monkeypatch.setattr(hosted, "generate_minimax_video", submit)
    node = hosted.MiniMaxH3VideoGenerationInvocation(prompt="animate")
    with pytest.raises(hosted.ExternalProviderRequestError, match="invalid MP4"):
        _run(invoker, node, owner)
    assert invoker.services.gallery.get_item_names(user_id=owner, media_kind="video").item_names == []


def test_queued_region_change_is_rejected_before_reading_gallery_images_or_submitting(
    invoker: Invoker, owner: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = hosted.MiniMaxH3VideoGenerationInvocation(
        prompt="animate", expected_region="global", first_frame=ImageField(image_name="private-frame.png")
    )
    invoker.services.configuration.update_config(
        {"external_minimax_base_url": "https://api.minimax.cn", "external_minimax_api_key": "cn-key"}
    )
    monkeypatch.setattr(ImagesInterface, "get_dto", lambda *args: pytest.fail("Read an image after the region changed"))
    monkeypatch.setattr(minimax_video, "_submit", lambda *args: pytest.fail("Submitted to a different region"))
    with pytest.raises(hosted.ExternalProviderNotConfiguredError, match="region changed"):
        _run(invoker, node, owner)


def test_rotation_after_execution_starts_preserves_the_key_and_destination_for_that_submission(
    invoker: Invoker, owner: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = invoker.services.configuration
    config.update_config({"external_minimax_api_key": "global-key", "external_minimax_base_url": None})
    node = hosted.MiniMaxH3VideoGenerationInvocation(prompt="animate", expected_region="global")

    def change_region(*args) -> None:
        config.update_config(
            {"external_minimax_api_key": "cn-key", "external_minimax_base_url": "https://api.minimax.cn"}
        )

    def capture_submission(session, origin, *args):
        assert origin == "https://api.minimax.io"
        assert session.headers["Authorization"] == "Bearer global-key"
        raise hosted.ExternalProviderRequestError("Captured test submission")

    monkeypatch.setattr(UtilInterface, "signal_progress", change_region)
    monkeypatch.setattr(minimax_video, "_submit", capture_submission)
    with pytest.raises(hosted.ExternalProviderRequestError, match="Captured test submission"):
        _run(invoker, node, owner)
    assert config.minimax_credentials() == ("cn-key", "https://api.minimax.cn")


def test_oversized_reference_prevents_any_pixel_decoding_or_submission(
    invoker: Invoker, owner: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = _image(invoker, owner, "blue")
    oversized = ImageField(image_name="oversized.png")
    invoker.services.image_records.save(
        image_name=oversized.image_name,
        image_origin=ResourceOrigin.EXTERNAL,
        image_category=ImageCategory.GENERAL,
        user_id=owner,
        width=6000,
        height=3000,
        has_workflow=False,
    )
    node = hosted.MiniMaxH3VideoGenerationInvocation(prompt="animate", reference_images=[valid, oversized])
    monkeypatch.setattr(
        ImagesInterface, "get_pil", lambda *args: pytest.fail("Decoded pixels before validating all sizes")
    )
    monkeypatch.setattr(minimax_video, "_submit", lambda *args: pytest.fail("Submitted oversized input"))
    with pytest.raises(ValueError, match="256–5760 pixels"):
        _run(invoker, node, owner)


@pytest.mark.parametrize("cancel_after", ["container", "thumbnail"])
def test_cancellation_skips_remaining_media_inspection_and_gallery_save(
    invoker: Invoker, owner: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cancel_after: str
) -> None:
    completed = []

    @contextmanager
    def download(**kwargs):
        yield tmp_path / "completed.mp4"

    def container(path):
        completed.append("container")
        return MediaProbe("h264", "aac")

    def thumbnail(path, **kwargs):
        completed.append("thumbnail")
        return (64, 48, 1, 8), None

    monkeypatch.setattr(hosted, "generate_minimax_video", download)
    monkeypatch.setattr(hosted, "is_mp4_file", lambda path: True)
    monkeypatch.setattr(hosted, "probe_media_streams", container)
    monkeypatch.setattr(hosted, "probe_decodable_video", thumbnail)
    monkeypatch.setattr(hosted, "decoder_frame_count", lambda path: pytest.fail("Counted frames after cancellation"))
    monkeypatch.setattr(UtilInterface, "is_canceled", lambda self: cancel_after in completed)
    with pytest.raises(hosted.CanceledException):
        _run(invoker, hosted.MiniMaxH3VideoGenerationInvocation(prompt="animate"), owner)
    assert completed[-1] == cancel_after
    assert invoker.services.gallery.get_item_names(user_id=owner, media_kind="video").total_count == 0
