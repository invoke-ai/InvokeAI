"""The upload routes announce gallery-visible uploads so open galleries refresh without polling."""

import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import videos
from invokeai.app.api_app import app
from invokeai.app.services.board_records.board_records_common import BoardRecord, BoardVisibility
from invokeai.app.services.events.events_common import ImageUploadedEvent, VideoUploadedEvent
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.videos.videos_common import VideoDTO


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


def _board(board_visibility: BoardVisibility) -> BoardRecord:
    return BoardRecord(
        board_id="board-1",
        board_name="Uploads",
        user_id="owner-1",
        created_at="None",
        updated_at="None",
        archived=False,
        board_visibility=board_visibility,
    )


def _image_dto(is_intermediate: bool) -> ImageDTO:
    return ImageDTO(
        image_name="upload.png",
        image_origin=ResourceOrigin.EXTERNAL,
        image_category=ImageCategory.USER,
        width=8,
        height=8,
        created_at="None",
        updated_at="None",
        is_intermediate=is_intermediate,
        starred=False,
        has_workflow=False,
        image_url="/images/upload.png",
        thumbnail_url="/thumbnails/upload.webp",
        board_id="board-1",
    )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _prepare_image_upload(monkeypatch: Any, mock_invoker: Invoker, is_intermediate: bool) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False
    monkeypatch.setattr(
        mock_invoker.services.board_records, "get", MagicMock(return_value=_board(BoardVisibility.Public))
    )
    monkeypatch.setattr(mock_invoker.services.images, "create", MagicMock(return_value=_image_dto(is_intermediate)))
    for target in (
        "invokeai.app.api.routers.images.ApiDependencies",
        "invokeai.app.api.routers._access.ApiDependencies",
        "invokeai.app.api.routers.image_move_maintenance.ApiDependencies",
        "invokeai.app.api.auth_dependencies.ApiDependencies",
    ):
        monkeypatch.setattr(target, mock_deps)


def _upload_events(mock_invoker: Invoker) -> list[ImageUploadedEvent | VideoUploadedEvent]:
    return [
        event
        for event in mock_invoker.services.events.events  # type: ignore[attr-defined]
        if isinstance(event, (ImageUploadedEvent, VideoUploadedEvent))
    ]


def test_image_upload_announces_the_upload_with_its_board_visibility(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=False)

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "user", "is_intermediate": False, "board_id": "board-1"},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    events = _upload_events(mock_invoker)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ImageUploadedEvent)
    assert event.image_name == "upload.png"
    assert event.image_category == ImageCategory.USER
    assert event.board_id == "board-1"
    assert event.board_owner_id == "owner-1"
    assert event.board_visibility == BoardVisibility.Public
    assert event.user_id == "system"


def test_image_upload_without_a_board_is_announced_with_no_board(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=False)
    mock_invoker.services.images.create.return_value = _image_dto(is_intermediate=False).model_copy(
        update={"board_id": None}
    )

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "user", "is_intermediate": False},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    mock_invoker.services.board_records.get.assert_not_called()
    events = _upload_events(mock_invoker)
    assert len(events) == 1
    assert events[0].board_id is None
    assert events[0].board_owner_id is None
    assert events[0].board_visibility is None


def test_failed_image_creation_is_a_mapped_500_with_no_announcement(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=False)
    mock_invoker.services.images.create.side_effect = RuntimeError("disk full")

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "user", "is_intermediate": False, "board_id": "board-1"},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to create image"
    assert _upload_events(mock_invoker) == []


@pytest.mark.parametrize(
    ("is_intermediate", "image_category"), [(True, "user"), (False, "other")], ids=["intermediate", "canvas-owned"]
)
def test_hidden_image_upload_is_not_announced(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, is_intermediate: bool, image_category: str
) -> None:
    """Canvas persistence uploads durable OTHER images on every paint flush, plus intermediates.

    Neither kind appears in a gallery view, so announcing them would refetch the gallery of the
    painter and of every admin for content nobody can see.
    """
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=is_intermediate)

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": image_category, "is_intermediate": is_intermediate, "board_id": "board-1"},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    assert _upload_events(mock_invoker) == []


def test_failed_board_attach_is_announced_without_the_board(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    """`create()` logs and continues when the board attach fails, so the DTO is the authority."""
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=False)
    mock_invoker.services.images.create.return_value = _image_dto(is_intermediate=False).model_copy(
        update={"board_id": None}
    )

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "user", "is_intermediate": False, "board_id": "board-1"},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    events = _upload_events(mock_invoker)
    assert len(events) == 1
    assert events[0].board_id is None
    assert events[0].board_owner_id is None
    assert events[0].board_visibility is None
    assert events[0].shared_user_ids == []


def test_private_board_upload_carries_its_share_recipients(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    """A private board's viewers are named on the event; the socket layer never queries storage."""
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=False)
    monkeypatch.setattr(
        mock_invoker.services.board_records, "get", MagicMock(return_value=_board(BoardVisibility.Private))
    )
    monkeypatch.setattr(
        mock_invoker.services.board_records, "get_shared_user_ids", MagicMock(return_value=["viewer-1", "viewer-2"])
    )

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "user", "is_intermediate": False, "board_id": "board-1"},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    events = _upload_events(mock_invoker)
    assert len(events) == 1
    assert events[0].shared_user_ids == ["viewer-1", "viewer-2"]


def test_upload_to_a_board_everyone_can_see_does_not_enumerate_shares(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    """Shared/Public uploads are broadcast, so paying for the share lookup would be waste."""
    _prepare_image_upload(monkeypatch, mock_invoker, is_intermediate=False)
    get_shared_user_ids = MagicMock(return_value=["viewer-1"])
    monkeypatch.setattr(mock_invoker.services.board_records, "get_shared_user_ids", get_shared_user_ids)

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "user", "is_intermediate": False, "board_id": "board-1"},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    get_shared_user_ids.assert_not_called()
    assert _upload_events(mock_invoker)[0].shared_user_ids == []


def _video_dto() -> VideoDTO:
    return VideoDTO(
        video_name="upload.mp4",
        video_origin=ResourceOrigin.EXTERNAL,
        video_category=ImageCategory.USER,
        width=8,
        height=8,
        duration=1.0,
        created_at="None",
        updated_at="None",
        is_intermediate=False,
        starred=False,
        has_workflow=False,
        video_url="/videos/upload.mp4",
        thumbnail_url="/thumbnails/upload.webp",
        board_id="board-1",
    )


def _prepare_video_upload(monkeypatch: Any, mock_invoker: Invoker) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr(mock_invoker.services, "videos", MagicMock(), raising=False)
    mock_invoker.services.videos.create.return_value = _video_dto()
    monkeypatch.setattr(
        mock_invoker.services.board_records, "get", MagicMock(return_value=_board(BoardVisibility.Shared))
    )
    for target in (
        "invokeai.app.api.routers.videos.ApiDependencies",
        "invokeai.app.api.routers._access.ApiDependencies",
        "invokeai.app.api.auth_dependencies.ApiDependencies",
        "invokeai.app.api_app.ApiDependencies",
    ):
        monkeypatch.setattr(target, mock_deps)
    # A bare `ftyp` box passes the container check; the codec and decode probes need ffmpeg.
    monkeypatch.setattr(
        videos, "probe_media_streams", lambda path: SimpleNamespace(video_codec="h264", audio_codec="aac")
    )
    monkeypatch.setattr(videos, "_probe_decodable_video", lambda path: ((8, 8, 1.0, 8.0), None))


def _post_video(client: TestClient, is_intermediate: bool, video_category: str = "user") -> Any:
    return client.post(
        "/api/v1/videos/upload",
        params={"video_category": video_category, "is_intermediate": is_intermediate, "board_id": "board-1"},
        files={"file": ("upload.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12, "video/mp4")},
    )


def test_video_upload_announces_the_upload_with_its_board_visibility(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    _prepare_video_upload(monkeypatch, mock_invoker)

    response = _post_video(client, is_intermediate=False)

    assert response.status_code == 201
    assert response.headers["location"] == "/videos/upload.mp4"
    events = _upload_events(mock_invoker)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, VideoUploadedEvent)
    assert event.video_name == "upload.mp4"
    assert event.board_id == "board-1"
    assert event.board_owner_id == "owner-1"
    assert event.board_visibility == BoardVisibility.Shared
    assert event.user_id == "system"


@pytest.mark.parametrize(
    ("is_intermediate", "video_category"), [(True, "user"), (False, "other")], ids=["intermediate", "canvas-owned"]
)
def test_hidden_video_upload_is_not_announced(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, is_intermediate: bool, video_category: str
) -> None:
    _prepare_video_upload(monkeypatch, mock_invoker)

    response = _post_video(client, is_intermediate=is_intermediate, video_category=video_category)

    assert response.status_code == 201
    assert _upload_events(mock_invoker) == []
