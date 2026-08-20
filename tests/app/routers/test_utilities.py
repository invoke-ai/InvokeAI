"""Router-level tests for /api/v1/utilities.

Covers:
- Auth gating (CurrentUserOrDefault on all three utility routes).
- image-to-prompt: image read-access check must fire BEFORE the model is loaded,
  so non-owners can't probe stored images.
- image-to-prompt: a missing image surfaces as 404, not 500.
"""

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from fontTools.ttLib import TTFont

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker


def _save_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=100,
        height=100,
        has_workflow=False,
        user_id=user_id,
    )


def _create_extra_user(mock_invoker: Invoker, email: str) -> str:
    from invokeai.app.services.users.users_common import UserCreateRequest

    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=email, password="TestPass123", is_admin=False)
    )
    return user.user_id


@pytest.fixture
def font_root(mock_invoker: Invoker, invokeai_root_dir: Path) -> Path:
    mock_invoker.services.configuration._root = invokeai_root_dir
    return invokeai_root_dir


# ----------------------------- Auth gating -----------------------------


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/v1/utilities/dynamicprompts", {"prompt": "hi"}),
        ("/api/v1/utilities/expand-prompt", {"prompt": "hi", "model_key": "m"}),
        ("/api/v1/utilities/image-to-prompt", {"image_name": "img-1", "model_key": "m"}),
    ],
)
def test_routes_require_auth(enable_multiuser: Any, client: TestClient, mock_invoker: Invoker, path: str, body: dict):
    r = client.post(path, json=body)
    assert r.status_code == status.HTTP_401_UNAUTHORIZED
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_dynamicprompts_works_for_user(client: TestClient, user1_token: str):
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a {b|c}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert "prompts" in body


def test_dynamicprompts_unknown_wildcard_returns_error_without_hanging(client: TestClient, user1_token: str):
    """An unknown wildcard used as a variant value would otherwise loop forever in the combinatorial generator.

    The endpoint must instead return promptly with a clear error and the original prompt echoed back.
    """
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__random__8chan|fenster|stuff}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is not None
    assert "random" in body["error"]
    assert body["prompts"] == ["{__random__8chan|fenster|stuff}"]


def test_dynamicprompts_bare_unknown_wildcard_still_generates(client: TestClient, user1_token: str):
    """A wildcard used as plain literal text (not a variant value) does not hang and must not error."""
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a photo, __my_style__"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is None
    assert body["prompts"]  # non-empty
    assert all(p == "a photo, __my_style__" for p in body["prompts"])


def test_dynamicprompts_random_generator_ignores_unknown_wildcard(client: TestClient, user1_token: str):
    """The random generator handles unknown wildcards gracefully, so the guard must not fire for it."""
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__random__8chan|fenster|stuff}", "combinatorial": False, "seed": 0},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is None
    assert body["prompts"]  # non-empty; the random generator expanded the variant


# ----------------------------- image_to_prompt: ownership / read-access -----------------------------


def test_image_to_prompt_forbidden_for_non_owner(
    client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
):
    """A second user must not be able to read a private image via image-to-prompt."""
    # Need to discover user1's id, then save an image under that id.
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "private-img.png", user1.user_id)

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "private-img.png", "model_key": "some-key"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    # The model must not have been loaded — ownership must fire BEFORE inference.
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_image_to_prompt_owner_reaches_model_load(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """The owner passes the read-access check and the model load is attempted.
    We force an UnknownModelException to keep the test light and assert 404."""
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "owned-img.png", user1.user_id)

    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no such model"))

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "owned-img.png", "model_key": "missing-model"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.model_manager.store.get_model.assert_called_once()


def test_image_to_prompt_admin_can_access_any_image(
    client: TestClient, admin_token: str, user1_token: str, mock_invoker: Invoker
):
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "user1-img.png", user1.user_id)

    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no model"))

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "user1-img.png", "model_key": "x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    # Admin passes the read-access check; model loading then fails with 404.
    assert r.status_code == status.HTTP_404_NOT_FOUND


def test_list_user_fonts_requires_auth(enable_multiuser: Any, font_root: Path, client: TestClient) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    r = client.get("/api/v1/utilities/fonts")

    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_user_font_file_requires_auth(enable_multiuser: Any, font_root: Path, client: TestClient) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    r = client.get("/api/v1/utilities/fonts/MyFont.ttf")

    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_user_fonts_support_real_font_files_and_configured_directory(
    admin_token: str, client: TestClient, font_root: Path, mock_invoker: Invoker
) -> None:
    assert mock_invoker.services.configuration.root_path == font_root
    mock_invoker.services.configuration.fonts_dir = Path("custom-fonts")
    fonts_dir = mock_invoker.services.configuration.fonts_path
    fonts_dir.mkdir(parents=True, exist_ok=True)
    source_font = Path(__file__).parents[3] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf"
    shutil.copyfile(source_font, fonts_dir / "Inter-Regular.ttf")

    r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert len(body["fonts"]) == 1
    assert body["fonts"][0]["family"] == "Inter"
    assert body["fonts"][0]["url"] == "api/v1/utilities/fonts/Inter-Regular.ttf"

    font_response = client.get(
        "/api/v1/utilities/fonts/Inter-Regular.ttf",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert font_response.status_code == status.HTTP_200_OK
    assert font_response.headers["content-type"] == "font/ttf"
    assert font_response.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert font_response.headers["content-disposition"].startswith('inline; filename="Inter-Regular.ttf"')
    assert font_response.content == source_font.read_bytes()


def test_list_user_fonts_reads_real_woff2_file(
    admin_token: str, client: TestClient, mock_invoker: Invoker, tmp_path: Path
) -> None:
    mock_invoker.services.configuration.fonts_dir = tmp_path / "fonts"
    fonts_dir = mock_invoker.services.configuration.fonts_path
    fonts_dir.mkdir(parents=True, exist_ok=True)
    source_font = Path(__file__).parents[3] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf"
    font = TTFont(source_font)
    try:
        font.flavor = "woff2"
        font.save(fonts_dir / "Inter-Regular.woff2")
    finally:
        font.close()

    r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert len(body["fonts"]) == 1
    assert body["fonts"][0]["family"] == "Inter"
    assert body["fonts"][0]["faces"][0]["path"] == "Inter-Regular.woff2"


def test_list_user_fonts_allows_authenticated_access(
    admin_token: str, client: TestClient, font_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")
    monkeypatch.setattr(
        "invokeai.app.api.routers.utilities._get_font_metadata",
        lambda _font_file: ("My Font", "My Font", 400, "normal"),
    )

    r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert len(body["fonts"]) == 1
    assert body["fonts"][0]["id"] == "user:my font"


def test_list_user_fonts_id_is_stable_when_preferred_face_changes(
    admin_token: str,
    client: TestClient,
    mock_invoker: Invoker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mock_invoker.services.configuration.fonts_dir = tmp_path / "fonts"
    fonts_dir = mock_invoker.services.configuration.fonts_path
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont-Bold.ttf").write_bytes(b"not-a-real-font")

    def get_metadata(font_file: Path) -> tuple[str, str, int, str]:
        weight = 400 if "Regular" in font_file.stem else 700
        return ("My Font", "My Font", weight, "normal")

    monkeypatch.setattr("invokeai.app.api.routers.utilities._get_font_metadata", get_metadata)

    first_response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})
    first_font = first_response.json()["fonts"][0]
    assert first_font["id"] == "user:my font"
    assert first_font["path"] == "MyFont-Bold.ttf"

    (fonts_dir / "MyFont-Regular.ttf").write_bytes(b"not-a-real-font")
    second_response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})
    second_font = second_response.json()["fonts"][0]
    assert second_font["id"] == first_font["id"]
    assert second_font["path"] == "MyFont-Regular.ttf"


def test_list_user_fonts_skips_malformed_fonts_and_logs_warning(
    admin_token: str,
    client: TestClient,
    font_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "BrokenFont.ttf").write_bytes(b"not-a-real-font")

    with caplog.at_level("WARNING"):
        r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    assert r.json()["fonts"] == []
    assert "Skipping font file" in caplog.text


def test_get_user_font_file_rejects_symlink(
    admin_token: str, client: TestClient, font_root: Path, tmp_path: Path
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.ttf"
    outside_file.write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked.ttf"

    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    r = client.get("/api/v1/utilities/fonts/linked.ttf", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_400_BAD_REQUEST


def test_list_user_fonts_skips_symlinked_files(
    admin_token: str, client: TestClient, font_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside-fonts"
    outside_dir.mkdir()
    (outside_dir / "outside.ttf").write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked-dir"

    try:
        symlink_path.symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    with caplog.at_level("WARNING"):
        r = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert r.status_code == status.HTTP_200_OK
    assert r.json()["fonts"] == []
    assert "Skipping font path" in caplog.text
