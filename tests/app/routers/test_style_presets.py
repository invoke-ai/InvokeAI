"""Router-level tests for /api/v1/style_presets.

Covers:
- Auth gating (CurrentUserOrDefault on CRUD/list/image, AdminUserOrDefault on export/import).
- Bug regression: json.JSONDecodeError must surface as 400 (not 500).
- Bug regression: in update_style_preset, a malformed `data` payload after a valid
  image upload must NOT have persisted the image (validation first, mutation second).
"""

import io
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetData,
    PresetType,
    StylePresetRecordDTO,
)


def _record(preset_id: str = "preset-1") -> StylePresetRecordDTO:
    return StylePresetRecordDTO(
        id=preset_id,
        name="Test",
        preset_data=PresetData(positive_prompt="p", negative_prompt="n"),
        type=PresetType.User,
    )


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------- Auth gating -----------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/style_presets/i/preset-1"),
        ("DELETE", "/api/v1/style_presets/i/preset-1"),
        ("GET", "/api/v1/style_presets/"),
        ("GET", "/api/v1/style_presets/i/preset-1/image"),
        ("GET", "/api/v1/style_presets/export"),
    ],
)
def test_simple_routes_require_auth(enable_multiuser: Any, client: TestClient, method: str, path: str):
    r = client.request(method, path)
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_create_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.post(
        "/api/v1/style_presets/",
        data={"data": json.dumps({"name": "x", "positive_prompt": "p", "negative_prompt": "n", "type": "user"})},
    )
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_update_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": json.dumps({"name": "x", "positive_prompt": "p", "negative_prompt": "n", "type": "user"})},
    )
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_import_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.post(
        "/api/v1/style_presets/import",
        files={"file": ("x.csv", b"name,prompt,negative_prompt\n", "text/csv")},
    )
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_export_forbidden_for_regular_user(client: TestClient, user1_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_records.get_many = MagicMock(return_value=[])
    r = client.get("/api/v1/style_presets/export", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_403_FORBIDDEN


def test_export_allowed_for_admin(client: TestClient, admin_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_records.get_many = MagicMock(return_value=[])
    r = client.get("/api/v1/style_presets/export", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == status.HTTP_200_OK


def test_import_forbidden_for_regular_user(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.post(
        "/api/v1/style_presets/import",
        files={"file": ("x.csv", b"name,prompt,negative_prompt\n", "text/csv")},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.style_preset_records.create_many.assert_not_called()


def test_list_allowed_for_regular_user(client: TestClient, user1_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_records.get_many = MagicMock(return_value=[])
    r = client.get("/api/v1/style_presets/", headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == []


# ----------------------------- Bug B regression: JSONDecodeError → 400 -----------------------------


def test_create_malformed_json_returns_400(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.post(
        "/api/v1/style_presets/",
        data={"data": "not-valid-json"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.style_preset_records.create.assert_not_called()


def test_update_malformed_json_returns_400(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": "not-valid-json"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.style_preset_records.update.assert_not_called()


# ----------------------------- Bug C regression: validation before image mutation -----------------------------


def test_update_with_invalid_data_does_not_save_image(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """A valid image plus malformed `data` must reject (400) AND must not have
    persisted or deleted the preset image — the validation has to run first."""
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": "not-valid-json"},
        files={"image": ("x.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.style_preset_records.update.assert_not_called()
    mock_invoker.services.style_preset_image_files.save.assert_not_called()
    mock_invoker.services.style_preset_image_files.delete.assert_not_called()


def test_update_without_image_and_invalid_data_does_not_delete(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    """The pre-fix code would call `delete` on the image even though `data` is invalid."""
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": "not-valid-json"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.style_preset_image_files.delete.assert_not_called()


# ----------------------------- Happy path: create + update -----------------------------


def test_create_with_valid_data_succeeds(client: TestClient, user1_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_records.create = MagicMock(return_value=_record("new-id"))
    mock_invoker.services.style_preset_image_files.get_url = MagicMock(return_value=None)
    r = client.post(
        "/api/v1/style_presets/",
        data={"data": json.dumps({"name": "Test", "positive_prompt": "p", "negative_prompt": "n", "type": "user"})},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    mock_invoker.services.style_preset_records.create.assert_called_once()


def test_update_with_valid_data_calls_update(client: TestClient, user1_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_records.update = MagicMock(return_value=_record("preset-1"))
    mock_invoker.services.style_preset_image_files.get_url = MagicMock(return_value=None)
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": json.dumps({"name": "Test", "positive_prompt": "p", "negative_prompt": "n", "type": "user"})},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    mock_invoker.services.style_preset_records.update.assert_called_once()


# ----------------------------- 415 for non-image upload is preserved -----------------------------


def test_update_with_non_image_returns_415(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": json.dumps({"name": "Test", "positive_prompt": "p", "negative_prompt": "n", "type": "user"})},
        files={"image": ("x.txt", b"not an image", "text/plain")},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    mock_invoker.services.style_preset_image_files.save.assert_not_called()
