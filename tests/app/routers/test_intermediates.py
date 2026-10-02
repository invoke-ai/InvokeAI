"""The intermediates API: authorization at the HTTP boundary and the preview → operation flow."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker


@pytest.fixture
def storage_ready(enable_multiuser: Any, mock_invoker: Invoker, tmp_path: Path) -> None:
    services = mock_invoker.services
    services.image_files = DiskImageFileStorage(tmp_path / "outputs")
    services.image_files.start(mock_invoker)
    services.image_moves = MagicMock()
    services.image_moves.is_maintenance_active.return_value = False
    services.videos = MagicMock()
    services.videos.delete_intermediates_by_names.return_value = MagicMock(deleted_names=[], purge_deferred=[])
    services.session_queue = MagicMock()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _user_id(mock_invoker: Invoker, email: str) -> str:
    user = mock_invoker.services.users.get_by_email(email)
    assert user is not None
    return user.user_id


def _seed_intermediate(mock_invoker: Invoker, name: str, user_id: str) -> None:
    mock_invoker.services.image_records.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        has_workflow=False,
        is_intermediate=True,
        user_id=user_id,
    )
    with mock_invoker.services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE images SET created_at = '2020-01-01 00:00:00.000' WHERE image_name = ?;", (name,))
    mock_invoker.services.image_files.save(image=Image.new("RGB", (8, 8)), image_name=name)


def _wait(client: TestClient, token: str, operation_id: str) -> dict:
    import time

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/intermediates/operations/{operation_id}", headers=_auth(token))
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("operation did not finish")


def test_every_route_requires_authentication(enable_multiuser: Any, client: TestClient) -> None:
    assert client.get("/api/v1/intermediates/summary").status_code == status.HTTP_401_UNAUTHORIZED
    assert (
        client.post("/api/v1/intermediates/previews", json={"mode": "safe", "scope": {"kind": "everyone"}}).status_code
        == status.HTTP_401_UNAUTHORIZED
    )
    assert client.get("/api/v1/intermediates/operations").status_code == status.HTTP_401_UNAUTHORIZED
    assert (
        client.post("/api/v1/intermediates/operations", json={"preview_id": "x"}).status_code
        == status.HTTP_401_UNAUTHORIZED
    )
    assert client.get("/api/v1/intermediates/operations/x").status_code == status.HTTP_401_UNAUTHORIZED
    assert client.put("/api/v1/intermediates/holds/tab", json={"images": ["x.png"]}).status_code == 401
    assert client.delete("/api/v1/intermediates/holds/tab").status_code == 401


def test_non_admins_see_only_their_rows_and_cannot_widen_scope(
    storage_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str, user2_token: str
) -> None:
    _seed_intermediate(mock_invoker, "u1.png", _user_id(mock_invoker, "user1@test.com"))
    _seed_intermediate(mock_invoker, "u2.png", _user_id(mock_invoker, "user2@test.com"))

    summary = client.get("/api/v1/intermediates/summary", headers=_auth(user1_token))
    assert summary.status_code == 200
    assert [row["user_id"] for row in summary.json()["items"]] == [_user_id(mock_invoker, "user1@test.com")]
    assert summary.json()["can_manage_everyone"] is False

    other = _user_id(mock_invoker, "user2@test.com")
    assert (
        client.get("/api/v1/intermediates/summary", params={"owner_id": other}, headers=_auth(user1_token)).status_code
        == status.HTTP_403_FORBIDDEN
    )
    for scope in ({"kind": "everyone"}, {"kind": "owner", "user_id": other}, {"kind": "matching", "user_id": other}):
        response = client.post(
            "/api/v1/intermediates/previews", json={"mode": "safe", "scope": scope}, headers=_auth(user1_token)
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN, response.text
    forged = client.post(
        "/api/v1/intermediates/previews",
        json={"mode": "safe", "scope": {"kind": "selection", "targets": [{"user_id": other, "project_id": None}]}},
        headers=_auth(user1_token),
    )
    assert forged.status_code == status.HTTP_403_FORBIDDEN
    for malformed in ({"kind": "owner"}, {"kind": "matching", "search": "no such project"}):
        response = client.post(
            "/api/v1/intermediates/previews", json={"mode": "safe", "scope": malformed}, headers=_auth(user1_token)
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text


def test_preview_operation_and_list_flow(
    storage_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str, user2_token: str, admin_token: str
) -> None:
    user1 = _user_id(mock_invoker, "user1@test.com")
    _seed_intermediate(mock_invoker, "u1.png", user1)

    preview = client.post(
        "/api/v1/intermediates/previews",
        json={"mode": "safe", "scope": {"kind": "matching", "excluded": []}},
        headers=_auth(user1_token),
    )
    assert preview.status_code == status.HTTP_201_CREATED, preview.text
    assert preview.json()["impact"]["delete_images"] == 1
    assert preview.json()["scope"]["kind"] == "matching"
    preview_id = preview.json()["preview_id"]

    stolen = client.post(
        "/api/v1/intermediates/operations", json={"preview_id": preview_id}, headers=_auth(user2_token)
    )
    assert stolen.status_code == status.HTTP_404_NOT_FOUND

    started = client.post(
        "/api/v1/intermediates/operations", json={"preview_id": preview_id}, headers=_auth(user1_token)
    )
    assert started.status_code == status.HTTP_202_ACCEPTED, started.text
    operation_id = started.json()["operation_id"]
    # A preview is confirmed once; a client whose response was lost finds the run in the list.
    repeated = client.post(
        "/api/v1/intermediates/operations", json={"preview_id": preview_id}, headers=_auth(user1_token)
    )
    assert repeated.status_code == status.HTTP_404_NOT_FOUND
    listed = client.get("/api/v1/intermediates/operations", headers=_auth(user1_token))
    assert [item["operation_id"] for item in listed.json()["items"]] == [operation_id]
    assert client.get("/api/v1/intermediates/operations", headers=_auth(user2_token)).json() == {"items": []}

    finished = _wait(client, user1_token, operation_id)
    assert finished["status"] == "completed"
    assert finished["progress"]["deleted_images"] == 1
    assert finished["scope"] == preview.json()["scope"]

    assert (
        client.get(f"/api/v1/intermediates/operations/{operation_id}", headers=_auth(user2_token)).status_code
        == status.HTTP_404_NOT_FOUND
    )
    assert client.get(f"/api/v1/intermediates/operations/{operation_id}", headers=_auth(admin_token)).status_code == 200


def test_mutations_are_refused_during_image_storage_maintenance(
    storage_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str
) -> None:
    user1 = _user_id(mock_invoker, "user1@test.com")
    preview = client.post(
        "/api/v1/intermediates/previews",
        json={"mode": "safe", "scope": {"kind": "owner", "user_id": user1}},
        headers=_auth(user1_token),
    )
    mock_invoker.services.image_moves.is_maintenance_active.return_value = True

    started = client.post(
        "/api/v1/intermediates/operations",
        json={"preview_id": preview.json()["preview_id"]},
        headers=_auth(user1_token),
    )
    assert started.status_code == status.HTTP_409_CONFLICT
    assert started.json()["detail"] == "Image storage maintenance is active"


def test_hold_is_account_scoped_at_the_http_boundary(
    storage_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str, user2_token: str
) -> None:
    user1 = _user_id(mock_invoker, "user1@test.com")
    _seed_intermediate(mock_invoker, "held.png", user1)
    assert (
        client.put(
            "/api/v1/intermediates/holds/tab", json={"images": ["held.png"]}, headers=_auth(user1_token)
        ).status_code
        == 204
    )
    assert client.delete("/api/v1/intermediates/holds/tab", headers=_auth(user2_token)).status_code == 204
    summary = client.get("/api/v1/intermediates/summary", headers=_auth(user1_token)).json()
    assert summary["items"][0]["images"]["active"] == 1


def test_hold_lease_ids_are_single_tokens(storage_ready: None, client: TestClient, user1_token: str) -> None:
    for lease_id in ("tab", "tab-1_x"):
        assert (
            client.put(f"/api/v1/intermediates/holds/{lease_id}", json={}, headers=_auth(user1_token)).status_code
            == 204
        )
    for lease_id in ("tab.0-0", ".0-0", "tab x", "tab/.."):
        assert (
            client.put(f"/api/v1/intermediates/holds/{lease_id}", json={}, headers=_auth(user1_token)).status_code
            != 204
        ), lease_id
        assert client.delete(f"/api/v1/intermediates/holds/{lease_id}", headers=_auth(user1_token)).status_code != 204


def test_legacy_clear_keeps_its_shape_and_the_safety_policy(
    storage_ready: None, mock_invoker: Invoker, client: TestClient, user1_token: str, admin_token: str
) -> None:
    user1 = _user_id(mock_invoker, "user1@test.com")
    _seed_intermediate(mock_invoker, "safe.png", user1)
    _seed_intermediate(mock_invoker, "referenced.png", user1)
    mock_invoker.services.project_records.create(user1, "P", {"imageName": "referenced.png"})

    # The count is what the clear would delete, so the legacy button settles at zero.
    assert client.get("/api/v1/images/intermediates", headers=_auth(user1_token)).json() == 1
    assert (
        client.delete("/api/v1/images/intermediates", headers=_auth(user1_token)).status_code
        == status.HTTP_403_FORBIDDEN
    )

    cleared = client.delete("/api/v1/images/intermediates", headers=_auth(admin_token))
    assert cleared.status_code == 200
    assert cleared.json() == 1
    assert client.get("/api/v1/images/intermediates", headers=_auth(user1_token)).json() == 0
    assert mock_invoker.services.image_records.get("referenced.png").is_intermediate
