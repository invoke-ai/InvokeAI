"""Router-level tests for /api/v1/style_presets.

Backed by a real SqliteStylePresetRecordsStorage from the shared conftest, so SQL
filtering and ownership rules are exercised end-to-end. style_preset_image_files
remains a MagicMock — file IO is not under test here.

Covers:
- Auth gating (CurrentUserOrDefault on CRUD/list/image, AdminUserOrDefault on export/import).
- Bug regression: json.JSONDecodeError must surface as 400 (not 500).
- Bug regression: malformed `data` payload after a valid image upload must NOT have
  persisted the image (validation runs before image mutation).
- Cross-user isolation: owner / non-owner / admin / default / public matrix on
  get, list, update, delete, and image fetch.
"""

import io
import json
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetData,
    PresetType,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    return buf.getvalue()


def _user_id(mock_invoker: Invoker, email: str) -> str:
    user = mock_invoker.services.users.get_by_email(email)
    assert user is not None, f"user {email} not seeded"
    return user.user_id


def _seed(
    mock_invoker: Invoker,
    user_id: str,
    name: str = "P",
    is_public: bool = False,
    preset_type: PresetType = PresetType.User,
) -> StylePresetRecordDTO:
    return mock_invoker.services.style_preset_records.create(
        StylePresetWithoutId(
            name=name,
            preset_data=PresetData(positive_prompt="p", negative_prompt="n"),
            type=preset_type,
            is_public=is_public,
        ),
        user_id=user_id,
    )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _form(name: str = "X", preset_type: str = "user", is_public: bool = False) -> dict[str, str]:
    return {
        "data": json.dumps(
            {
                "name": name,
                "positive_prompt": "p",
                "negative_prompt": "n",
                "type": preset_type,
                "is_public": is_public,
            }
        )
    }


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
    r = client.post("/api/v1/style_presets/", data=_form())
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_update_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.patch("/api/v1/style_presets/i/preset-1", data=_form())
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_import_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.post(
        "/api/v1/style_presets/import",
        files={"file": ("x.csv", b"name,prompt,negative_prompt\n", "text/csv")},
    )
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_export_forbidden_for_regular_user(client: TestClient, user1_token: str):
    r = client.get("/api/v1/style_presets/export", headers=_auth(user1_token))
    assert r.status_code == status.HTTP_403_FORBIDDEN


def test_export_allowed_for_admin(client: TestClient, admin_token: str):
    r = client.get("/api/v1/style_presets/export", headers=_auth(admin_token))
    assert r.status_code == status.HTTP_200_OK


def test_import_forbidden_for_regular_user(client: TestClient, user1_token: str):
    r = client.post(
        "/api/v1/style_presets/import",
        files={"file": ("x.csv", b"name,prompt,negative_prompt\n", "text/csv")},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN


# ----------------------------- Bug B regression: JSONDecodeError → 400 -----------------------------


def test_create_malformed_json_returns_400(client: TestClient, user1_token: str, mock_invoker: Invoker):
    r = client.post("/api/v1/style_presets/", data={"data": "not-valid-json"}, headers=_auth(user1_token))
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    # Nothing persisted.
    assert mock_invoker.services.style_preset_records.get_many(user_id=_user_id(mock_invoker, "user1@test.com")) == []


def test_update_malformed_json_returns_400(client: TestClient, user1_token: str, mock_invoker: Invoker):
    # No need for a real record — JSON validation happens before the record is loaded.
    r = client.patch("/api/v1/style_presets/i/some-id", data={"data": "not-valid-json"}, headers=_auth(user1_token))
    assert r.status_code == status.HTTP_400_BAD_REQUEST


# ----------------------------- Bug C regression: validation before image mutation -----------------------------


def test_update_with_invalid_data_does_not_save_image(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """A valid image plus malformed `data` must reject (400) AND must not have
    persisted or deleted the preset image — the validation has to run first."""
    r = client.patch(
        "/api/v1/style_presets/i/preset-1",
        data={"data": "not-valid-json"},
        files={"image": ("x.png", _png_bytes(), "image/png")},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.style_preset_image_files.save.assert_not_called()
    mock_invoker.services.style_preset_image_files.delete.assert_not_called()


def test_update_without_image_and_invalid_data_does_not_delete(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    """The pre-fix code would call `delete` on the image even though `data` is invalid."""
    r = client.patch("/api/v1/style_presets/i/preset-1", data={"data": "not-valid-json"}, headers=_auth(user1_token))
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.style_preset_image_files.delete.assert_not_called()


# ----------------------------- Happy path -----------------------------


def test_create_with_valid_data_persists_record(client: TestClient, user1_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_image_files.get_url.return_value = None
    r = client.post("/api/v1/style_presets/", data=_form(name="Mine"), headers=_auth(user1_token))
    assert r.status_code == status.HTTP_200_OK
    user1 = _user_id(mock_invoker, "user1@test.com")
    presets = mock_invoker.services.style_preset_records.get_many(user_id=user1)
    names = [p.name for p in presets]
    assert "Mine" in names
    # New record is owned by user1.
    owned = [p for p in presets if p.name == "Mine"]
    assert owned[0].user_id == user1
    assert owned[0].is_public is False


def test_update_with_valid_data_changes_record(client: TestClient, user1_token: str, mock_invoker: Invoker):
    mock_invoker.services.style_preset_image_files.get_url.return_value = None
    user1 = _user_id(mock_invoker, "user1@test.com")
    seeded = _seed(mock_invoker, user1, name="Before")

    r = client.patch(
        f"/api/v1/style_presets/i/{seeded.id}",
        data=_form(name="After"),
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_200_OK
    refreshed = mock_invoker.services.style_preset_records.get(seeded.id)
    assert refreshed.name == "After"


def test_update_with_non_image_returns_415(client: TestClient, user1_token: str, mock_invoker: Invoker):
    user1 = _user_id(mock_invoker, "user1@test.com")
    seeded = _seed(mock_invoker, user1, name="X")
    r = client.patch(
        f"/api/v1/style_presets/i/{seeded.id}",
        data=_form(name="X"),
        files={"image": ("x.txt", b"not an image", "text/plain")},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    mock_invoker.services.style_preset_image_files.save.assert_not_called()


# ----------------------------- Cross-user ownership policy -----------------------------


class TestOwnership:
    def test_list_returns_only_own_plus_defaults_plus_public(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        u2 = _user_id(mock_invoker, "user2@test.com")
        _seed(mock_invoker, u1, name="u1-private")
        _seed(mock_invoker, u1, name="u1-public", is_public=True)
        _seed(mock_invoker, u2, name="u2-private")
        _seed(mock_invoker, "system", name="default-A", preset_type=PresetType.Default)

        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get("/api/v1/style_presets/", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_200_OK
        names = {p["name"] for p in r.json()}
        assert "u1-private" in names
        assert "u1-public" in names
        assert "default-A" in names
        assert "u2-private" not in names

    def test_list_includes_other_users_public_preset(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        _seed(mock_invoker, u1, name="u1-public", is_public=True)
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get("/api/v1/style_presets/", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_200_OK
        names = {p["name"] for p in r.json()}
        assert "u1-public" in names

    def test_get_other_users_private_is_forbidden(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="private")
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_get_other_users_public_is_allowed(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="pub", is_public=True)
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_200_OK
        assert r.json()["name"] == "pub"

    def test_get_default_is_allowed_for_any_user(self, client: TestClient, user1_token: str, mock_invoker: Invoker):
        seeded = _seed(mock_invoker, "system", name="builtin", preset_type=PresetType.Default)
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_200_OK

    def test_update_other_users_preset_is_forbidden(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="orig")
        r = client.patch(
            f"/api/v1/style_presets/i/{seeded.id}",
            data=_form(name="hijack"),
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN
        # Even a public preset cannot be modified by a non-owner.
        seeded_public = _seed(mock_invoker, u1, name="orig-pub", is_public=True)
        r = client.patch(
            f"/api/v1/style_presets/i/{seeded_public.id}",
            data=_form(name="hijack-pub"),
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_other_users_preset_is_forbidden(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="keep")
        r = client.delete(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN
        # Record is still there.
        still = mock_invoker.services.style_preset_records.get(seeded.id)
        assert still.id == seeded.id

    def test_update_default_preset_forbidden_for_non_admin(
        self, client: TestClient, user1_token: str, mock_invoker: Invoker
    ):
        seeded = _seed(mock_invoker, "system", name="builtin", preset_type=PresetType.Default)
        r = client.patch(
            f"/api/v1/style_presets/i/{seeded.id}",
            data=_form(name="hijacked", preset_type="default"),
            headers=_auth(user1_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_default_preset_forbidden_for_non_admin(
        self, client: TestClient, user1_token: str, mock_invoker: Invoker
    ):
        seeded = _seed(mock_invoker, "system", name="builtin", preset_type=PresetType.Default)
        r = client.delete(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_create_default_preset_forbidden_for_non_admin(
        self, client: TestClient, user1_token: str, mock_invoker: Invoker
    ):
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.post(
            "/api/v1/style_presets/",
            data=_form(name="Sneaky", preset_type="default"),
            headers=_auth(user1_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_admin_can_get_any_preset(
        self, client: TestClient, admin_token: str, user1_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="private")
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(admin_token))
        assert r.status_code == status.HTTP_200_OK

    def test_admin_can_update_any_preset(
        self, client: TestClient, admin_token: str, user1_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="orig")
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.patch(
            f"/api/v1/style_presets/i/{seeded.id}",
            data=_form(name="admin-edit"),
            headers=_auth(admin_token),
        )
        assert r.status_code == status.HTTP_200_OK
        assert mock_invoker.services.style_preset_records.get(seeded.id).name == "admin-edit"

    def test_admin_can_delete_default_preset(self, client: TestClient, admin_token: str, mock_invoker: Invoker):
        seeded = _seed(mock_invoker, "system", name="del-default", preset_type=PresetType.Default)
        r = client.delete(f"/api/v1/style_presets/i/{seeded.id}", headers=_auth(admin_token))
        # delete returns 200 with no body (operation_id has no explicit status_code)
        assert r.status_code in (status.HTTP_200_OK, status.HTTP_204_NO_CONTENT)

    def test_admin_list_returns_everything(
        self, client: TestClient, admin_token: str, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        u2 = _user_id(mock_invoker, "user2@test.com")
        _seed(mock_invoker, u1, name="u1-only")
        _seed(mock_invoker, u2, name="u2-only")
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        r = client.get("/api/v1/style_presets/", headers=_auth(admin_token))
        names = {p["name"] for p in r.json()}
        assert {"u1-only", "u2-only"}.issubset(names)

    def test_owner_can_flip_to_public(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="will-be-public")
        mock_invoker.services.style_preset_image_files.get_url.return_value = None
        # user2 can't see it yet
        r = client.get("/api/v1/style_presets/", headers=_auth(user2_token))
        assert "will-be-public" not in {p["name"] for p in r.json()}

        # user1 flips is_public=True
        r = client.patch(
            f"/api/v1/style_presets/i/{seeded.id}",
            data=_form(name="will-be-public", is_public=True),
            headers=_auth(user1_token),
        )
        assert r.status_code == status.HTTP_200_OK
        assert r.json()["is_public"] is True

        # user2 now sees it
        r = client.get("/api/v1/style_presets/", headers=_auth(user2_token))
        assert "will-be-public" in {p["name"] for p in r.json()}

    def test_image_fetch_enforces_same_policy_as_get(
        self, client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
    ):
        u1 = _user_id(mock_invoker, "user1@test.com")
        seeded = _seed(mock_invoker, u1, name="img-private")
        r = client.get(f"/api/v1/style_presets/i/{seeded.id}/image", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN
        mock_invoker.services.style_preset_image_files.get_path.assert_not_called()
