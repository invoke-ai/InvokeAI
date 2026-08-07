"""Tests for the server-side media copy routes.

Duplicating a project cannot share its media: `board_images` and `board_videos` key on the media
name, so one item sits on exactly one board. These routes are what makes that copy cheap — the
bytes never leave the server — so what matters here is that a copy is faithful (category, origin
and provenance travel; intermediates and starring do not), lands where it was told to, and that
one bad name cannot cost the caller the batch.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.urls.urls_default import LocalUrlService

SOURCE_METADATA = {"positive_prompt": "a cat", "seed": 12345}
SOURCE_WORKFLOW = '{"name": "a workflow"}'


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


@pytest.fixture
def real_images(mock_invoker: Invoker, tmp_path: Path) -> DiskImageFileStorage:
    """Wire the real image file storage and name service behind the real `ImageService`.

    Deliberately not a MagicMock. What this route has to get right is what happens *to the files*:
    that the copy is a new file, that the source keeps its own, and that provenance survives. A
    stubbed service asserts only that the route called something, which is how a copy that
    consumed its source would pass a full suite.
    """
    image_files = DiskImageFileStorage(tmp_path / "images")
    mock_invoker.services.image_files = image_files
    mock_invoker.services.names = SimpleNameService()
    mock_invoker.services.urls = LocalUrlService()
    image_files.start(mock_invoker)
    mock_invoker.services.images.start(mock_invoker)
    return image_files


def _create_source_image(
    mock_invoker: Invoker,
    user_id: str,
    category: ImageCategory = ImageCategory.CONTROL,
) -> ImageDTO:
    """A genuinely stored image: record, file, thumbnail and embedded provenance."""
    return mock_invoker.services.images.create(
        image=Image.new("RGB", (8, 8), "red"),
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        metadata=json.dumps(SOURCE_METADATA),
        workflow=SOURCE_WORKFLOW,
        user_id=user_id,
    )


def _copy_images(client: TestClient, token: str, **body: Any):
    return client.post("/api/v1/images/copy", json=body, headers=_auth(token))


def test_copying_an_image_preserves_its_category_origin_and_metadata(
    client: TestClient, mock_invoker: Invoker, user1_token: str, real_images: DiskImageFileStorage
):
    user_id = _owner_id(client, user1_token)
    board_id = _create_board(client, user1_token)
    source = _create_source_image(mock_invoker, user_id)

    response = _copy_images(client, user1_token, image_names=[source.image_name], board_id=board_id)

    assert response.status_code == status.HTTP_200_OK
    copy_name = response.json()["copied"][0]["image_name"]
    assert response.json()["failed"] == []
    assert copy_name != source.image_name

    copy_dto = mock_invoker.services.images.get_dto(copy_name)
    assert copy_dto.image_category == ImageCategory.CONTROL
    assert copy_dto.image_origin == ResourceOrigin.INTERNAL
    assert copy_dto.board_id == board_id
    assert copy_dto.is_intermediate is False
    assert copy_dto.starred is False
    assert mock_invoker.services.image_records.get_user_id(copy_name) == user_id
    # Provenance rides in the PNG's chunks and the record; both must arrive.
    metadata = mock_invoker.services.images.get_metadata(copy_name)
    assert metadata is not None and json.loads(metadata.model_dump_json()) == SOURCE_METADATA
    assert mock_invoker.services.images.get_workflow(copy_name) == SOURCE_WORKFLOW
    # Neither the originating session nor node travels.
    record = mock_invoker.services.images.get_record(copy_name)
    assert record.session_id is None
    assert record.node_id is None


def test_copying_an_image_leaves_the_source_untouched(
    client: TestClient, mock_invoker: Invoker, user1_token: str, real_images: DiskImageFileStorage
):
    """The copy must be additive. A copy that consumes or rewrites its source destroys the
    original project when its duplicate is made."""
    user_id = _owner_id(client, user1_token)
    source = _create_source_image(mock_invoker, user_id)
    source_path = real_images.get_path(source.image_name, image_subfolder=source.image_subfolder)
    source_bytes = source_path.read_bytes()

    response = _copy_images(client, user1_token, image_names=[source.image_name])
    copy_name = response.json()["copied"][0]["image_name"]

    assert source_path.exists()
    assert source_path.read_bytes() == source_bytes
    # The copy is its own file, byte-identical rather than re-encoded.
    copy_record = mock_invoker.services.images.get_record(copy_name)
    copy_path = real_images.get_path(copy_name, image_subfolder=copy_record.image_subfolder)
    assert copy_path != source_path
    assert copy_path.read_bytes() == source_bytes
    # And the source's own provenance still reads back the way it did before the copy.
    assert mock_invoker.services.images.get_workflow(source.image_name) == SOURCE_WORKFLOW
    source_metadata = mock_invoker.services.images.get_metadata(source.image_name)
    assert source_metadata is not None and json.loads(source_metadata.model_dump_json()) == SOURCE_METADATA


def test_copying_without_a_board_leaves_the_copy_uncategorized(
    client: TestClient, mock_invoker: Invoker, user1_token: str, real_images: DiskImageFileStorage
):
    user_id = _owner_id(client, user1_token)
    source = _create_source_image(mock_invoker, user_id)

    response = _copy_images(client, user1_token, image_names=[source.image_name])

    assert response.status_code == status.HTTP_200_OK
    copy_name = response.json()["copied"][0]["image_name"]
    assert mock_invoker.services.images.get_dto(copy_name).board_id is None


def test_one_unreadable_source_does_not_cost_the_batch(
    client: TestClient, mock_invoker: Invoker, user1_token: str, real_images: DiskImageFileStorage
):
    user_id = _owner_id(client, user1_token)
    source = _create_source_image(mock_invoker, user_id)

    response = _copy_images(client, user1_token, image_names=[source.image_name, "missing.png"])

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert [entry["source_image_name"] for entry in body["copied"]] == [source.image_name]
    assert body["failed"] == ["missing.png"]


def test_a_failed_copy_leaves_no_ghost_record(
    client: TestClient, mock_invoker: Invoker, user1_token: str, real_images: DiskImageFileStorage
):
    """A record whose file never arrived is worse than no copy: the gallery shows a tile whose
    every endpoint 404s, and nothing points at it to clean it up."""
    user_id = _owner_id(client, user1_token)
    source = _create_source_image(mock_invoker, user_id)
    before = set(mock_invoker.services.image_records.get_image_names().image_names)

    # The file is gone but the record remains — a source the copy cannot materialize.
    real_images.get_path(source.image_name, image_subfolder=source.image_subfolder).unlink()

    response = _copy_images(client, user1_token, image_names=[source.image_name])

    assert response.json() == {"copied": [], "failed": [source.image_name]}
    assert set(mock_invoker.services.image_records.get_image_names().image_names) == before


def test_copying_into_a_board_you_cannot_write_is_refused(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str, real_images: DiskImageFileStorage
):
    theirs = _create_board(client, user2_token, "Theirs")
    user_id = _owner_id(client, user1_token)
    source = _create_source_image(mock_invoker, user_id)
    before = set(mock_invoker.services.image_records.get_image_names().image_names)

    refused = _copy_images(client, user1_token, image_names=[source.image_name], board_id=theirs)
    missing = _copy_images(client, user1_token, image_names=[source.image_name], board_id="no-such-board")

    assert refused.status_code == status.HTTP_403_FORBIDDEN
    assert missing.status_code == status.HTTP_404_NOT_FOUND
    # The refusal lands before anything is copied.
    assert set(mock_invoker.services.image_records.get_image_names().image_names) == before


def test_copying_someone_elses_image_is_refused_per_name(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str, real_images: DiskImageFileStorage
):
    mine = _owner_id(client, user1_token)
    theirs = _owner_id(client, user2_token)
    source = _create_source_image(mock_invoker, mine)
    _insert_image_record(mock_invoker, "theirs.png", theirs)

    response = _copy_images(client, user1_token, image_names=[source.image_name, "theirs.png"])

    body = response.json()
    assert [entry["source_image_name"] for entry in body["copied"]] == [source.image_name]
    assert body["failed"] == ["theirs.png"]


def test_a_batch_larger_than_the_cap_is_refused(
    client: TestClient, mock_invoker: Invoker, user1_token: str, real_images: DiskImageFileStorage
):
    """Each name is a record insert and a file copy the server performs synchronously, so an
    uncapped batch is an unbounded amount of work one request can ask for."""
    _owner_id(client, user1_token)

    response = _copy_images(client, user1_token, image_names=[f"img-{index}.png" for index in range(1001)])

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


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
    # The route checks the copy actually landed where it was told, so the double has to say so.
    created.board_id = board_id
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
    # The path handed over is the source's own managed file, and `create` consumes what it is
    # given. Without this the copy moves the original's bytes away from it.
    assert kwargs["move_source"] is False


def test_a_video_copy_that_missed_its_board_is_reported_as_failed(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    """`create` treats board attachment as best-effort, which is right for a generation and wrong
    here: the caller is about to remap a document onto the name we return."""
    user_id = _owner_id(client, user1_token)
    board_id = _create_board(client, user1_token, "Video+Target")
    _insert_video_record(mock_invoker, "src.mp4", user_id)

    videos = MagicMock()
    videos.get_path.return_value = "/tmp/source.mp4"
    videos.get_metadata.return_value = None
    videos.get_workflow.return_value = None
    videos.get_graph.return_value = None
    created = MagicMock()
    created.video_name = "copy-001.mp4"
    created.board_id = None  # The attachment silently did not happen.
    videos.create.return_value = created
    mock_invoker.services.videos = videos

    response = client.post(
        "/api/v1/videos/copy",
        json={"video_names": ["src.mp4"], "board_id": board_id},
        headers=_auth(user1_token),
    )

    assert response.json() == {"copied": [], "failed": ["src.mp4"]}


def test_copying_someone_elses_video_is_refused_per_name(client: TestClient, mock_invoker: Invoker, user1_token: str):
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
