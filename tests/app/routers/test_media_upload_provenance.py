"""Uploads record which of the caller's own projects they originate in, and nothing else."""

import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.videos.videos_common import VideoDTO
from invokeai.app.util.video_ingest import MediaProbe


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_dto() -> ImageDTO:
    return ImageDTO(
        image_name="upload.png",
        image_origin=ResourceOrigin.EXTERNAL,
        image_category=ImageCategory.USER,
        width=8,
        height=8,
        created_at="None",
        updated_at="None",
        is_intermediate=True,
        starred=False,
        has_workflow=False,
        image_url="/images/upload.png",
        thumbnail_url="/thumbnails/upload.webp",
    )


@pytest.fixture
def upload_ready(enable_multiuser: Any, mock_invoker: Invoker, monkeypatch: Any) -> None:
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False
    monkeypatch.setattr(mock_invoker.services.images, "create", MagicMock(return_value=_image_dto()))
    mock_invoker.services.videos = MagicMock()


def _own_project_id(mock_invoker: Invoker, client: TestClient, token: str) -> str:
    response = client.post("/api/v1/projects/", json={"name": "Mine", "data": {}}, headers=_auth(token))
    assert response.status_code == 201
    return response.json()["project_id"]


def test_image_upload_records_the_callers_project(
    upload_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str
) -> None:
    project_id = _own_project_id(mock_invoker, client, user1_token)

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "other", "is_intermediate": True, "project_id": project_id},
        files={"file": ("upload.png", _png_bytes(), "image/png")},
        headers=_auth(user1_token),
    )

    assert response.status_code == 201
    assert mock_invoker.services.images.create.call_args.kwargs["project_id"] == project_id


def test_image_upload_refuses_a_project_the_caller_does_not_own(
    upload_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str, user2_token: str, admin_token: str
) -> None:
    user2s = _own_project_id(mock_invoker, client, user2_token)

    # Even an admin cannot attribute an upload to someone else's project.
    for token, project_id in ((user1_token, user2s), (admin_token, user2s), (admin_token, "no-such-project")):
        response = client.post(
            "/api/v1/images/upload",
            params={"image_category": "other", "is_intermediate": True, "project_id": project_id},
            files={"file": ("upload.png", _png_bytes(), "image/png")},
            headers=_auth(token),
        )
        assert response.status_code == 404, response.text
        # The webv2 upload fallback matches this exact detail.
        assert response.json()["detail"] == "Project not found"

    mock_invoker.services.images.create.assert_not_called()


def test_video_upload_validates_the_project_before_reading_the_body(
    upload_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str
) -> None:
    with patch("invokeai.app.api.routers.videos._stream_video_upload") as stream:
        response = client.post(
            "/api/v1/videos/upload",
            params={"video_category": "general", "is_intermediate": True, "project_id": "not-mine"},
            files={"file": ("video.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12, "video/mp4")},
            headers=_auth(user1_token),
        )

    assert response.status_code == 404
    stream.assert_not_called()
    mock_invoker.services.videos.create.assert_not_called()


def test_video_upload_records_the_callers_project(
    upload_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str
) -> None:
    project_id = _own_project_id(mock_invoker, client, user1_token)
    mock_invoker.services.videos.create.return_value = VideoDTO.model_validate(
        {
            "video_name": "uploaded.mp4",
            "video_origin": "external",
            "video_category": "general",
            "width": 64,
            "height": 64,
            "duration": 1.0,
            "fps": 8.0,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "is_intermediate": True,
            "starred": False,
            "has_workflow": False,
            "video_subfolder": "",
            "video_url": "/api/v1/videos/i/uploaded.mp4/full",
            "thumbnail_url": "/api/v1/videos/i/uploaded.mp4/thumbnail",
        }
    )

    with (
        patch(
            "invokeai.app.api.routers.videos.probe_media_streams",
            return_value=MediaProbe(video_codec="h264", audio_codec="aac"),
        ),
        patch("invokeai.app.api.routers.videos._probe_decodable_video", return_value=((64, 64, 1.0, 8.0), None)),
    ):
        response = client.post(
            "/api/v1/videos/upload",
            params={"video_category": "general", "is_intermediate": True, "project_id": project_id},
            files={"file": ("video.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12, "video/mp4")},
            headers=_auth(user1_token),
        )

    assert response.status_code == 201, response.text
    assert mock_invoker.services.videos.create.call_args.kwargs["project_id"] == project_id
