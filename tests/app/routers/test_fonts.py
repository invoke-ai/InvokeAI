"""HTTP coverage for the authenticated custom-font catalog."""

from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.fonts.fonts_default import FontService
from invokeai.app.services.invoker import Invoker


@pytest.fixture
def font_service(mock_invoker: Invoker, tmp_path: Path) -> FontService:
    service = FontService(
        db=mock_invoker.services.image_records._db,
        fonts_dir=tmp_path / "fonts",
        storage_dir=tmp_path / "uploaded-fonts",
    )
    mock_invoker.services.fonts = service
    yield service
    service.stop()


@pytest.fixture
def font_bytes() -> bytes:
    return (Path(__file__).parents[3] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf").read_bytes()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _upload(client: TestClient, token: str, data: bytes, filename: str = "Inter-Regular.ttf", **form: str):
    return client.post(
        "/api/v1/fonts",
        headers=_auth(token),
        files={"file": (filename, data, "font/ttf")},
        data=form,
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/api/v1/fonts"), ("post", "/api/v1/fonts/validate")],
)
def test_font_routes_require_auth(
    method: str, path: str, enable_multiuser: Any, client: TestClient, font_bytes: bytes
) -> None:
    request = getattr(client, method)
    kwargs = {"files": {"file": ("font.ttf", font_bytes, "font/ttf")}} if method == "post" else {}
    response = request(path, **kwargs)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_upload_download_validate_and_hash_filter(
    client: TestClient, user1_token: str, font_service: FontService, font_bytes: bytes
) -> None:
    validation = client.post(
        "/api/v1/fonts/validate",
        headers=_auth(user1_token),
        files={"file": ("preview.ttf", font_bytes, "font/ttf")},
    )
    assert validation.status_code == status.HTTP_200_OK
    assert validation.json()["content_hash"]
    assert font_service.list(user_id="system")[1] == 0

    response = _upload(client, user1_token, font_bytes, filename="Résumé.ttf")
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["created"] is True
    font = body["font"]
    assert font["source"] == "uploaded"
    assert font["scope"] == "private"
    assert font["url"].endswith("/file")

    duplicate = _upload(client, user1_token, font_bytes, filename="renamed.ttf")
    assert duplicate.status_code == status.HTTP_200_OK
    assert duplicate.json()["created"] is False
    assert duplicate.json()["font"]["id"] == font["id"]

    filtered = client.get(
        "/api/v1/fonts",
        params={"content_hash": font["content_hash"]},
        headers=_auth(user1_token),
    )
    assert filtered.status_code == status.HTTP_200_OK
    assert filtered.json()["total"] == 1
    assert filtered.headers["cache-control"] == "private, no-store"

    metadata = client.get(f"/api/v1/fonts/{font['id']}", headers=_auth(user1_token))
    assert metadata.status_code == status.HTTP_200_OK
    assert metadata.headers["cache-control"] == "private, no-store"

    downloaded = client.get(font["url"], headers=_auth(user1_token))
    assert downloaded.status_code == status.HTTP_200_OK
    assert downloaded.content == font_bytes
    assert downloaded.headers["content-type"] == "font/ttf"
    assert downloaded.headers["cache-control"] == "private, no-store"
    assert downloaded.headers["content-disposition"] == "inline; filename*=UTF-8''R%C3%A9sum%C3%A9.ttf"

    changed = client.get(font["url"], params={"expected_hash": "0" * 64}, headers=_auth(user1_token))
    assert changed.status_code == status.HTTP_409_CONFLICT


def test_private_font_is_hidden_from_other_users_and_owner_can_delete(
    client: TestClient, user1_token: str, user2_token: str, font_service: FontService, font_bytes: bytes
) -> None:
    uploaded = _upload(client, user1_token, font_bytes).json()["font"]

    listed = client.get("/api/v1/fonts", headers=_auth(user2_token))
    assert listed.status_code == status.HTTP_200_OK
    assert all(item["id"] != uploaded["id"] for item in listed.json()["items"])

    hidden = client.get(uploaded["url"], headers=_auth(user2_token))
    assert hidden.status_code == status.HTTP_404_NOT_FOUND

    deleted = client.delete(f"/api/v1/fonts/{uploaded['id']}", headers=_auth(user1_token))
    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    assert font_service.get(uploaded["id"]) is None
    assert not list(font_service._storage_dir.glob("*"))


def test_shared_upload_and_admin_rescan_authorization(
    client: TestClient,
    admin_token: str,
    user1_token: str,
    user2_token: str,
    font_service: FontService,
    font_bytes: bytes,
) -> None:
    regular_shared = _upload(client, user1_token, font_bytes, scope="shared")
    assert regular_shared.status_code == status.HTTP_403_FORBIDDEN

    shared = _upload(client, admin_token, font_bytes, filename="Shared.ttf", scope="shared")
    assert shared.status_code == status.HTTP_200_OK
    shared_font = shared.json()["font"]
    listed = client.get("/api/v1/fonts", headers=_auth(user2_token))
    assert any(item["id"] == shared_font["id"] for item in listed.json()["items"])

    forbidden_delete = client.delete(f"/api/v1/fonts/{shared_font['id']}", headers=_auth(user2_token))
    assert forbidden_delete.status_code == status.HTTP_403_FORBIDDEN

    directory_file = font_service._fonts_dir / "Directory.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    rescan = client.post("/api/v1/fonts/rescan", headers=_auth(user1_token))
    assert rescan.status_code == status.HTTP_403_FORBIDDEN
    rescan = client.post("/api/v1/fonts/rescan", headers=_auth(admin_token))
    assert rescan.status_code == status.HTTP_200_OK
    assert rescan.json()["indexed"] == 1

    deleted = client.delete(f"/api/v1/fonts/{shared_font['id']}", headers=_auth(admin_token))
    assert deleted.status_code == status.HTTP_204_NO_CONTENT


def test_static_instance_route_returns_original_binary(
    client: TestClient, user1_token: str, font_service: FontService, font_bytes: bytes
) -> None:
    uploaded = _upload(client, user1_token, font_bytes).json()["font"]
    response = client.post(
        f"/api/v1/fonts/{uploaded['id']}/instance",
        headers=_auth(user1_token),
        json={"content_hash": uploaded["content_hash"], "coordinates": {}},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.content == font_bytes
    assert response.headers["content-type"] == "font/ttf"
    assert response.headers["cache-control"] == "private, no-store"
