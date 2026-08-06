"""Tests for the server-side media copy routes.

Duplicating a project cannot share its media: `board_images` and `board_videos` key on the media
name, so one item sits on exactly one board. These routes are what makes that copy cheap — the
bytes never leave the server — so what matters here is that a copy is faithful (category, origin
and provenance travel; intermediates and starring do not), lands where it was told to, and that
one bad name cannot cost the caller the batch.
"""

import json
from typing import Any
from unittest.mock import MagicMock

from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker

SOURCE_METADATA = {"positive_prompt": "a cat", "seed": 12345}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_board(client: TestClient, token: str, name: str = "Target") -> str:
    response = client.post(f"/api/v1/boards/?board_name={name}", headers=_auth(token))
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()["board_id"]


def _owner_id(client: TestClient, token: str) -> str:
    board = client.post("/api/v1/boards/?board_name=Probe", headers=_auth(token))
    return board.json()["user_id"]


def _insert_image_record(mock_invoker: Invoker, name: str, user_id: str, category: str = "control") -> None:
    """A real row, so the route's read-access check runs against real data."""
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO images (image_name, image_origin, image_category, width, height, user_id)"
            " VALUES (?, 'internal', ?, 64, 64, ?);",
            (name, category, user_id),
        )


def _pil_with_metadata() -> Image.Image:
    """A PIL image carrying the text chunks `image_files_disk.save` writes."""
    image = Image.new("RGB", (8, 8))
    image.info = {"invokeai_metadata": json.dumps(SOURCE_METADATA)}
    return image


def _stub_images_service(mock_invoker: Invoker, record_category: ImageCategory = ImageCategory.CONTROL) -> MagicMock:
    """Replace the images service, keeping `image_records` real for the access check."""
    images = MagicMock()
    record = MagicMock()
    record.image_origin = ResourceOrigin.INTERNAL
    record.image_category = record_category
    images.get_record.return_value = record
    images.get_pil_image.return_value = _pil_with_metadata()
    created = MagicMock()
    created.image_name = "copy-001.png"
    images.create.return_value = created
    mock_invoker.services.images = images
    return images


def _copy_images(client: TestClient, token: str, **body: Any):
    return client.post("/api/v1/images/copy", json=body, headers=_auth(token))


def test_copying_an_image_preserves_its_category_origin_and_metadata(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user_id = _owner_id(client, user1_token)
    board_id = _create_board(client, user1_token)
    _insert_image_record(mock_invoker, "src.png", user_id)
    images = _stub_images_service(mock_invoker)

    response = _copy_images(client, user1_token, image_names=["src.png"], board_id=board_id)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"copied": [{"source_image_name": "src.png", "image_name": "copy-001.png"}], "failed": []}

    kwargs = images.create.call_args.kwargs
    assert kwargs["image_category"] == ImageCategory.CONTROL
    assert kwargs["image_origin"] == ResourceOrigin.INTERNAL
    assert kwargs["board_id"] == board_id
    assert kwargs["is_intermediate"] is False
    assert kwargs["user_id"] == user_id
    # Provenance rides in the PNG's text chunks; the copy must read it back the way upload does.
    assert json.loads(kwargs["metadata"]) == SOURCE_METADATA
    # Neither the originating session nor node travels.
    assert kwargs.get("session_id") is None
    assert kwargs.get("node_id") is None


def test_copying_without_a_board_leaves_the_copy_uncategorized(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user_id = _owner_id(client, user1_token)
    _insert_image_record(mock_invoker, "src.png", user_id)
    images = _stub_images_service(mock_invoker)

    response = _copy_images(client, user1_token, image_names=["src.png"])

    assert response.status_code == status.HTTP_200_OK
    assert images.create.call_args.kwargs["board_id"] is None


def test_one_unreadable_source_does_not_cost_the_batch(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user_id = _owner_id(client, user1_token)
    _insert_image_record(mock_invoker, "good.png", user_id)
    images = _stub_images_service(mock_invoker)

    response = _copy_images(client, user1_token, image_names=["good.png", "missing.png"])

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert [entry["source_image_name"] for entry in body["copied"]] == ["good.png"]
    assert body["failed"] == ["missing.png"]
    assert images.create.call_count == 1


def test_copying_into_a_board_you_cannot_write_is_refused(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
):
    theirs = _create_board(client, user2_token, "Theirs")
    user_id = _owner_id(client, user1_token)
    _insert_image_record(mock_invoker, "src.png", user_id)
    images = _stub_images_service(mock_invoker)

    refused = _copy_images(client, user1_token, image_names=["src.png"], board_id=theirs)
    missing = _copy_images(client, user1_token, image_names=["src.png"], board_id="no-such-board")

    assert refused.status_code == status.HTTP_403_FORBIDDEN
    assert missing.status_code == status.HTTP_404_NOT_FOUND
    # The refusal lands before anything is copied.
    images.create.assert_not_called()


def test_copying_someone_elses_image_is_refused_per_name(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
):
    mine = _owner_id(client, user1_token)
    theirs = _owner_id(client, user2_token)
    _insert_image_record(mock_invoker, "mine.png", mine)
    _insert_image_record(mock_invoker, "theirs.png", theirs)
    _stub_images_service(mock_invoker)

    response = _copy_images(client, user1_token, image_names=["mine.png", "theirs.png"])

    body = response.json()
    assert [entry["source_image_name"] for entry in body["copied"]] == ["mine.png"]
    assert body["failed"] == ["theirs.png"]


def _insert_video_record(mock_invoker: Invoker, name: str, user_id: str) -> None:
    """A real row, so the board service's cover resolution keeps working around it."""
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO videos (video_name, video_origin, video_category, width, height, duration, fps, user_id)"
            " VALUES (?, 'internal', 'general', 640, 480, 2.5, 24.0, ?);",
            (name, user_id),
        )


def test_copying_a_video_preserves_its_shape_and_provenance(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user_id = _owner_id(client, user1_token)
    board_id = _create_board(client, user1_token, "Video+Target")
    _insert_video_record(mock_invoker, "src.mp4", user_id)

    videos = MagicMock()
    videos.get_path.return_value = "/tmp/source.mp4"
    metadata = MagicMock()
    metadata.model_dump_json.return_value = json.dumps(SOURCE_METADATA)
    videos.get_metadata.return_value = metadata
    videos.get_workflow.return_value = "{}"
    videos.get_graph.return_value = None
    created = MagicMock()
    created.video_name = "copy-001.mp4"
    videos.create.return_value = created
    mock_invoker.services.videos = videos

    response = client.post(
        "/api/v1/videos/copy",
        json={"video_names": ["src.mp4"], "board_id": board_id},
        headers=_auth(user1_token),
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"copied": [{"source_video_name": "src.mp4", "video_name": "copy-001.mp4"}], "failed": []}
    kwargs = videos.create.call_args.kwargs
    assert (kwargs["width"], kwargs["height"], kwargs["duration"], kwargs["fps"]) == (640, 480, 2.5, 24.0)
    assert kwargs["video_category"] == ImageCategory.GENERAL
    assert kwargs["board_id"] == board_id
    assert kwargs["is_intermediate"] is False
    assert json.loads(kwargs["metadata"]) == SOURCE_METADATA
    assert kwargs["workflow"] == "{}"


def test_copying_someone_elses_video_is_refused_per_name(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    _owner_id(client, user1_token)
    _insert_video_record(mock_invoker, "theirs.mp4", "somebody-else")
    videos = MagicMock()
    mock_invoker.services.videos = videos

    response = client.post(
        "/api/v1/videos/copy",
        json={"video_names": ["theirs.mp4"]},
        headers=_auth(user1_token),
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"copied": [], "failed": ["theirs.mp4"]}
    videos.create.assert_not_called()
