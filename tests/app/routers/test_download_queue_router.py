"""Router-level tests for /api/v1/download_queue.

Covers:
- Auth gating. Every route is AdminUserOrDefault: the queue is a single server-wide
  queue whose jobs expose remote URLs and local filesystem paths, and cancelling one
  affects whoever started it.
- Bug regression: `dest` path validation must reject absolute paths and '..' segments
  BEFORE the queue service is invoked.
- Security regression: `dest` is anchored inside a separate download directory
  (never the model cache or process working directory) and `source` is rejected
  when it resolves to a non-public address.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.routers import download_queue as download_queue_router
from invokeai.app.api_app import app
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
def test_routes_require_auth_in_multiuser_mode(enable_multiuser: Any, client: TestClient, method: str, path: str):
    response = client.request(method, path, json={"source": "http://x/y", "dest": "models/x"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


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
def test_routes_forbidden_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker, method: str, path: str
):
    """No download_queue route is reachable by a non-admin in multiuser mode."""
    r = client.request(
        method,
        path,
        json={"source": "http://example.com/file.bin", "dest": "x"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.download_queue.download.assert_not_called()


def test_list_downloads_as_admin(client: TestClient, admin_token: str, mock_invoker: Invoker):
    mock_invoker.services.download_queue.list_jobs = MagicMock(return_value=[])
    r = client.get("/api/v1/download_queue/", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == []


def test_prune_downloads_forbidden_for_regular_user(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.patch("/api/v1/download_queue/", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.download_queue.prune_jobs.assert_not_called()


def test_prune_downloads_allowed_for_admin(client: TestClient, admin_token: str, mock_invoker: Invoker):
    r = client.patch("/api/v1/download_queue/", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == status.HTTP_204_NO_CONTENT
    mock_invoker.services.download_queue.prune_jobs.assert_called_once()


def test_cancel_all_forbidden_for_regular_user(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.delete("/api/v1/download_queue/i", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.download_queue.cancel_all_jobs.assert_not_called()


def test_cancel_all_allowed_for_admin(client: TestClient, admin_token: str, mock_invoker: Invoker):
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
        "models/x\x00.bin",
    ],
)
def test_download_rejects_unsafe_dest_before_service_call(
    client: TestClient, admin_token: str, mock_invoker: Invoker, bad_dest: str
):
    """Absolute paths, '..' segments, and empty strings must produce 400 and
    must NOT invoke the download_queue service."""
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://example.com/file.bin", "dest": bad_dest},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.download_queue.download.assert_not_called()


def test_download_accepts_relative_dest(client: TestClient, admin_token: str, mock_invoker: Invoker):
    cache_root = mock_invoker.services.configuration.download_cache_path.resolve()
    queue_root = cache_root.parent / f"{cache_root.name}.downloads"
    mock_invoker.services.download_queue.download = MagicMock(
        return_value=DownloadJob(
            id=1,
            source="http://example.com/file.bin",
            dest=queue_root / "models/sd15.safetensors",
        )
    )
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://example.com/file.bin", "dest": "models/sd15.safetensors"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    assert r.json()["dest"] == "models/sd15.safetensors"
    mock_invoker.services.download_queue.download.assert_called_once()


@pytest.mark.anyio
async def test_download_validation_does_not_block_event_loop(
    enable_multiuser: Any, admin_token: str, mock_invoker: Invoker, monkeypatch: Any, tmp_path: Path
):
    """Slow path and URL validation must not stall unrelated async work."""
    mock_invoker.services.configuration.download_cache_dir = tmp_path / "model-cache"
    mock_invoker.services.download_queue.download = MagicMock(return_value=_make_job())
    dest_started = threading.Event()
    url_started = threading.Event()
    dest_release = threading.Event()
    url_release = threading.Event()

    def slow_dest(*args: Any, **kwargs: Any) -> Path:
        dest_started.set()
        assert dest_release.wait(timeout=2)
        return tmp_path / "model-cache.downloads" / "file.bin"

    def slow_url(*args: Any, **kwargs: Any) -> None:
        url_started.set()
        assert url_release.wait(timeout=2)

    monkeypatch.setattr(download_queue_router, "_validate_dest", slow_dest)
    monkeypatch.setattr(download_queue_router, "validate_download_url", slow_url)

    async def assert_event_loop_runs() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/api/v1/app/version")
            assert response.status_code == status.HTTP_200_OK

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        task = asyncio.create_task(
            client.post(
                "/api/v1/download_queue/i/",
                json={"source": "http://example.com/file.bin", "dest": "file.bin"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        )
        try:
            assert await asyncio.to_thread(dest_started.wait, 1)
            await assert_event_loop_runs()
            dest_release.set()
            assert await asyncio.to_thread(url_started.wait, 1)
            await assert_event_loop_runs()
            url_release.set()
            response = await asyncio.wait_for(task, timeout=2)
            assert response.status_code == status.HTTP_200_OK
        finally:
            dest_release.set()
            url_release.set()


# ------------------- Advisory regression: dest confinement + SSRF -------------------


def test_dest_is_anchored_in_download_cache_not_cwd(
    client: TestClient, admin_token: str, mock_invoker: Invoker, monkeypatch: Any, tmp_path: Path
):
    """A relative `dest` must resolve under the API download directory, not the model cache or CWD.

    Before the fix, `dest="nodes/pwn/__init__.py"` landed in the working directory --
    which holds the custom-nodes directory in a source install and the application's own
    Python package in the container image.
    """
    cwd = tmp_path / "server_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    mock_invoker.services.download_queue.download = MagicMock(return_value=_make_job())
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://example.com/file.bin", "dest": "nodes/pwn/__init__.py"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_200_OK

    passed_dest = mock_invoker.services.download_queue.download.call_args.args[1]
    cache_root = mock_invoker.services.configuration.download_cache_path.resolve()
    queue_root = cache_root.parent / f"{cache_root.name}.downloads"
    assert passed_dest.is_absolute()
    assert passed_dest.is_relative_to(queue_root)
    assert not passed_dest.is_relative_to(cache_root)
    assert not passed_dest.is_relative_to(cwd)


def test_download_dest_cannot_poison_model_cache(
    client: TestClient, admin_token: str, mock_invoker: Invoker, tmp_path: Path
):
    """An arbitrary admin download must not land in the model cache consumed by inference."""
    mock_invoker.services.configuration.download_cache_dir = tmp_path / "model-cache"
    mock_invoker.services.download_queue.download = MagicMock(return_value=_make_job())
    model_cache = mock_invoker.services.configuration.download_cache_path.resolve()
    model_cache.mkdir(parents=True)
    (model_cache / "https-example-com-model").mkdir()

    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://example.com/model", "dest": "https-example-com-model/payload.pt"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert r.status_code == status.HTTP_200_OK, r.text
    passed_dest = mock_invoker.services.download_queue.download.call_args.args[1]
    assert passed_dest.parent != model_cache / "https-example-com-model"
    assert passed_dest.is_relative_to(model_cache.parent / "model-cache.downloads")


@pytest.mark.parametrize(
    "unsafe_source",
    [
        "http://127.0.0.1:19191/proof.txt",
        "http://localhost:19191/proof.txt",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://[::1]:8080/proof.txt",
        "http://0177.0.0.1/proof.txt",  # octal literal for 127.0.0.1
        "http://2130706433/proof.txt",  # integer literal for 127.0.0.1
    ],
)
def test_download_rejects_non_public_sources(
    client: TestClient, admin_token: str, mock_invoker: Invoker, unsafe_source: str
):
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": unsafe_source, "dest": "models/x.bin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST, r.text
    assert r.json()["detail"] == "Download URL resolves to a non-public address."
    mock_invoker.services.download_queue.download.assert_not_called()


def test_download_allows_non_public_sources_when_opted_in(client: TestClient, admin_token: str, mock_invoker: Invoker):
    """Operators with a model mirror on their LAN can opt back in."""
    mock_invoker.services.configuration.allow_private_download_urls = True
    mock_invoker.services.download_queue.download = MagicMock(return_value=_make_job())
    r = client.post(
        "/api/v1/download_queue/i/",
        json={"source": "http://10.0.0.5/mirror/sd15.safetensors", "dest": "models/x.bin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    mock_invoker.services.download_queue.download.assert_called_once()
