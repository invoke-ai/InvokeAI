"""Router-level tests for /api/v1/files.

Managed-file inputs upload through this router, so its registration is part of the feature:
a missing include makes every upload answer 405 from the SPA catch-all instead of reaching
the service. These tests go through the HTTP layer against a real `DiskFileService`.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.files.files_disk import DiskFileService
from invokeai.app.services.invoker import Invoker

UPLOAD_URL = "/api/v1/files/upload"
MAX_FILE_SIZE_BYTES = 1024


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def file_service(enable_multiuser: Any, mock_invoker: Invoker, tmp_path: Path) -> DiskFileService:
    service = DiskFileService(tmp_path / "files", max_file_size=MAX_FILE_SIZE_BYTES)
    mock_invoker.services.files = service
    return service


def _upload(
    client: TestClient,
    token: str,
    *,
    file_name: str = "notes.txt",
    content: bytes = b"hello",
    content_type: str = "text/plain",
):
    return client.post(
        UPLOAD_URL,
        files={"file": (file_name, content, content_type)},
        headers=_auth(token),
    )


def test_upload_get_delete_round_trip(client: TestClient, user1_token: str, file_service: DiskFileService) -> None:
    del file_service

    uploaded = _upload(client, user1_token)

    assert uploaded.status_code == status.HTTP_201_CREATED
    file = uploaded.json()
    assert (file["file_name"], file["content_type"], file["size_bytes"]) == ("notes.txt", "text/plain", 5)

    fetched = client.get(f"/api/v1/files/i/{file['file_id']}", headers=_auth(user1_token))
    assert fetched.status_code == status.HTTP_200_OK
    assert fetched.json() == file

    deleted = client.delete(f"/api/v1/files/i/{file['file_id']}", headers=_auth(user1_token))
    assert deleted.status_code == status.HTTP_204_NO_CONTENT

    gone = client.get(f"/api/v1/files/i/{file['file_id']}", headers=_auth(user1_token))
    assert gone.status_code == status.HTTP_404_NOT_FOUND


def test_another_user_cannot_read_or_delete_a_file(
    client: TestClient, user1_token: str, user2_token: str, file_service: DiskFileService
) -> None:
    del file_service
    file_id = _upload(client, user1_token).json()["file_id"]

    assert client.get(f"/api/v1/files/i/{file_id}", headers=_auth(user2_token)).status_code == status.HTTP_403_FORBIDDEN
    assert (
        client.delete(f"/api/v1/files/i/{file_id}", headers=_auth(user2_token)).status_code == status.HTTP_403_FORBIDDEN
    )
    # The refused delete must not have touched the owner's file.
    assert client.get(f"/api/v1/files/i/{file_id}", headers=_auth(user1_token)).status_code == status.HTTP_200_OK


def test_unsupported_extension_is_rejected(client: TestClient, user1_token: str, file_service: DiskFileService) -> None:
    del file_service

    response = _upload(client, user1_token, file_name="payload.exe", content_type="application/octet-stream")

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_oversized_file_is_rejected(client: TestClient, user1_token: str, file_service: DiskFileService) -> None:
    del file_service

    response = _upload(client, user1_token, content=b"x" * (MAX_FILE_SIZE_BYTES + 1))

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE


def test_exhausted_storage_quota_is_rejected(
    client: TestClient, user1_token: str, mock_invoker: Invoker, tmp_path: Path
) -> None:
    storage_path = tmp_path / "quota-files"
    mock_invoker.services.files = DiskFileService(storage_path, max_storage_size=8)

    assert _upload(client, user1_token, content=b"x" * 6).status_code == status.HTTP_201_CREATED

    response = _upload(client, user1_token, content=b"x" * 6)

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    # The rejected upload's partial file and metadata are cleaned up.
    assert len(list(storage_path.iterdir())) == 2
