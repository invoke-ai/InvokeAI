"""Router-level tests for /api/v1/download_queue.

Covers:
- Auth gating (CurrentUserOrDefault on read/per-job, AdminUserOrDefault on prune & cancel-all).
- Bug regression: `dest` path validation must reject absolute paths and '..' segments
  BEFORE the queue service is invoked.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.download import DownloadJob
from invokeai.app.services.invoker import Invoker


def _make_job(id: int = 1) -> DownloadJob:
    from pathlib import Path

    return DownloadJob(id=id, source="http://example.com/file.bin", dest=Path("models/file.bin"))


# ----------------------------- Auth gating -----------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/download_queue/"),
        ("PATCH", "/api/v1/download_queue/"),
        ("POST", "/api/v1/download_queue/i/"),
        ("GET", "/api/v1/download_queue/i/1"),
        ("DELETE", "/api/v1/download_queue/i/1"),
        ("DELETE", "/api/v1/download_queue/i"),
    ],
)
def test_routes_require_auth_in_multiuser_mode(
    enable_multiuser: Any, client: TestClient, method: str, path: str
):
    response = client.request(method, path, json={"source": "http://x/y", "dest": "models/x"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_downloads_as_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    mock_invoker.services.download_queue.list_jobs = MagicMock(return_value=[])
    r = client.get("/api/v1/download_queue/", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == []


def test_prune_downloads_forbidden_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    r = client.patch("/api/v1/download_queue/", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.download_queue.prune_jobs.assert_not_called()


def test_prune_downloads_allowed_for_admin(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    r = client.patch("/api/v1/download_queue/", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == status.HTTP_204_NO_CONTENT
    mock_invoker.services.download_queue.prune_jobs.assert_called_once()


def test_cancel_all_forbidden_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    r = client.delete("/api/v1/download_queue/i", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.download_queue.cancel_all_jobs.assert_not_called()


def test_cancel_all_allowed_for_admin(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    r = client.delete("/api/v1/download_queue/i", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == status.HTTP_204_NO_CONTENT
    mock_invoker.services.download_queue.cancel_all_jobs.assert_called_once()


# ----------------------------- Bug D regression: dest validation -----------------------------


@pytest.mark.parametrize(
    "bad_dest",
    [
        "/etc/passwd",
        "C:/Windows/System32",
        "models/../../etc/passwd",
        "..",
        "",
        "   ",
    ],
)
def test_download_rejects_unsafe_dest_before_service_call(
    client: TestClient, user1_token: str, mock_invoker: Invoker, bad_dest: str
):
    """Absolute paths, '..' segments, and empty strings must produce 400 and
    must NOT invoke the download_queue service."""
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://example.com/file.bin", "dest": bad_dest},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.download_queue.download.assert_not_called()


def test_download_accepts_relative_dest(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    mock_invoker.services.download_queue.download = MagicMock(return_value=_make_job())
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://example.com/file.bin", "dest": "models/sd15.safetensors"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    mock_invoker.services.download_queue.download.assert_called_once()
